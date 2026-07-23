import inspect

from src.llm.client import BaseLLMClient, LLMResponse
from src.tools.base import ToolRegistry
from langfuse import observe, get_client
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReactAgent:
    def __init__(self,llm_client:BaseLLMClient,tool_registry:ToolRegistry,system_prompt: str = ""):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt
        self.max_iterations = 100
        self.conversation_history = []
        self._image_turns = []  # [(history_index, [filenames])] not yet pruned

    def _prune_old_images(self) -> None:
        """Keep only the most recently shown turn's images as real inlineData -
        every older image is collapsed to a text placeholder so conversation
        history doesn't resend every image ever generated on every call."""
        while len(self._image_turns) > 1:
            old_index, old_filenames = self._image_turns.pop(0)
            old_parts = self.conversation_history[old_index]["parts"]
            new_parts = []
            filename_iter = iter(old_filenames)
            for part in old_parts:
                if "inlineData" in part:
                    new_parts.append({"text": f"[Image previously shown: {next(filename_iter)}]"})
                else:
                    new_parts.append(part)
            self.conversation_history[old_index]["parts"] = new_parts

    @observe()
    async def run(self, query: str, context: list[str] | None = None) -> str:
        logger.info(f"Start ReAct loop | query={query}")
        langfuse = get_client()
        
        if context is None:
            context = []
        
        if context:
            memory_text = "\n".join(f"- {item}" for item in context)
            full_system_prompt = f"{self.system_prompt}\n\nRelevant memory:\n{memory_text}"
        else:
            full_system_prompt = self.system_prompt
        
        self.conversation_history.append({"role": "user", "parts": [{"text": query}]})

        # The modeling-brief completion gate only applies to agents that
        # actually register that tool - a generic ReactAgent (e.g. a plain
        # search agent) has no way to satisfy it and would be nudged forever.
        requires_complete_brief = "finalize_modeling_brief" in self.tool_registry.tools
        brief_complete = not requires_complete_brief
        nudge_count = 0
        max_nudges = 3

        for iteration in range(self.max_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")
            response = await self.llm_client.send_message(
                message="",
                system_prompt=full_system_prompt,
                tools=self.tool_registry.get_all_schemas(),
                conversation_history=self.conversation_history
            )

            if response.tool_name:
                logger.info(f"Tool call | tool={response.tool_name} | args={response.tool_args}")
                tool = self.tool_registry.get_tool(response.tool_name)

                with langfuse.start_as_current_observation(
                    as_type="span",
                    name=response.tool_name,
                    input=response.tool_args,
                ) as span:
                    if tool is None:
                        tool_result = (
                            f"Tool call failed: no tool named '{response.tool_name}' "
                            f"exists. Check the tool name against your available tools."
                        )
                        logger.warning(f"Unknown tool requested | tool={response.tool_name}")
                    else:
                        try:
                            if inspect.iscoroutinefunction(tool.func):
                                tool_result = await tool.func(**response.tool_args)
                            else:
                                tool_result = tool.func(**response.tool_args)
                        except Exception as e:
                            tool_result = (
                                f"Tool call failed: {e}. Check that you provided all "
                                f"required arguments for {response.tool_name}."
                            )
                            logger.warning(
                                f"Tool call error | tool={response.tool_name} | error={e}"
                            )

                    if isinstance(tool_result, dict):
                        text_result = tool_result.get("text", "")
                        images = tool_result.get("images", [])
                    else:
                        text_result = tool_result
                        images = []

                    span.update(output=text_result)

                logger.info(f"Tool result received | tool={response.tool_name} | images={len(images)}")

                if response.tool_name == "finalize_modeling_brief" and tool is not None:
                    brief_complete = isinstance(tool_result, dict) and bool(tool_result.get("complete"))

                self.conversation_history.append({
                    "role": "model",
                    "parts": response.raw_parts
                })
                parts = [{"functionResponse": {"name": response.tool_name, "response": {"output": text_result}}}]
                parts.extend(
                    {"inlineData": {"mimeType": "image/png", "data": img["data"]}} for img in images
                )
                turn_index = len(self.conversation_history)
                self.conversation_history.append({"role": "user", "parts": parts})

                if images:
                    self._image_turns.append((turn_index, [img["filename"] for img in images]))
                    self._prune_old_images()

            elif response.finish_reason == "STOP":
                if not brief_complete and nudge_count < max_nudges:
                    nudge_count += 1
                    logger.info(
                        f"Agent tried to stop without a complete modeling brief | "
                        f"nudging ({nudge_count}/{max_nudges})"
                    )
                    self.conversation_history.append({
                        "role": "model",
                        "parts": response.raw_parts
                    })
                    self.conversation_history.append({
                        "role": "user",
                        "parts": [{"text": (
                            "You have not finalized a complete modeling brief yet. "
                            "Before concluding, call finalize_modeling_brief with all "
                            "required fields - if you already tried and it told you "
                            "what was missing, address that and call it again."
                        )}]
                    })
                    continue

                logger.info(f"Agent finished | iterations={iteration + 1}")
                self.conversation_history.append({
                    "role": "model",
                    "parts": response.raw_parts
                })
                return response.content

        return "Max iterations reached without a final answer"