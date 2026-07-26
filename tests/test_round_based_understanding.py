from src.agent import round_based_understanding as rbu
from src.llm.gemini_embeddings import GeminiEmbeddingClient
from src.memory.manager import KnowledgeStore, ModelingBriefStore, UnderstandingRoundStore


def _stub_schema(monkeypatch, columns, row_count=1000, entity_columns=None):
    monkeypatch.setattr(rbu, "fetch_table_columns", lambda table_name: columns)
    monkeypatch.setattr(rbu, "fetch_row_count", lambda table_name: row_count)
    monkeypatch.setattr(rbu, "detect_entity_columns", lambda table_name, cols, rc: entity_columns or [])


def _full_preprocessing_rules():
    return [
        {"category": "missing_values", "rule": "none needed", "detail": "d"},
        {"category": "placeholder_values", "rule": "none needed", "detail": "d"},
        {"category": "scaling", "rule": "none needed", "detail": "d"},
        {"category": "encoding", "rule": "none needed", "detail": "d"},
    ]


_NO_ANCHOR = {"has_anchor": False, "reason": "checked, nothing found"}


async def test_round_based_understanding_converges_after_quiet_streak(db_session, dataset_id, monkeypatch):
    embedding_client = GeminiEmbeddingClient()
    brief_store = ModelingBriefStore(db_session)
    knowledge_store = KnowledgeStore(embedding_client, db_session)

    _stub_schema(monkeypatch, columns=[
        {"name": "wind_speed", "type": "double precision"},
        {"name": "rotor_temp", "type": "double precision"},
    ])

    async def fake_round(llm_client, embedding_client_, db_session_, table_name, ds_id, project_brief, round_number, query_log, blind_findings=None):
        # Only round 1 does anything - rounds 2+ simulate a genuinely quiet
        # round (nothing left to find), which is what should trigger
        # convergence after CONVERGENCE_STREAK quiet rounds in a row.
        if round_number == 1:
            brief_store.save_brief(
                dataset_id=ds_id,
                label_status="absent",
                label_column=None,
                label_notes="no label found",
                feature_set=[
                    {"column": "wind_speed", "role": "feature", "reason": "r"},
                    {"column": "rotor_temp", "role": "feature", "reason": "r"},
                ],
                preprocessing_rules=_full_preprocessing_rules(),
                validation_strategy="v",
                confidence=0.8,
                validation_anchor=_NO_ANCHOR
            )
            await knowledge_store.save_observation(ds_id, "wind_speed correlates with rotor_temp", 0.8)
            query_log.append('SELECT COUNT(*) FROM "t"')

    monkeypatch.setattr(rbu, "_run_one_round", fake_round)

    result = await rbu.run_round_based_understanding(
        llm_client=None,
        embedding_client=embedding_client,
        db_session=db_session,
        table_name="turbine_data",
        dataset_id=dataset_id,
        project_brief="test brief",
        max_rounds=10
    )

    assert result["converged"] is True
    assert result["rounds_run"] == 3  # round 1 (progress) + 2 consecutive quiet rounds
    assert result["blind_cycles"] == 1  # blind check agrees immediately, no reconciliation needed

    round_store = UnderstandingRoundStore(db_session)
    rounds = round_store.get_rounds(dataset_id)
    assert len(rounds) == 3
    assert rounds[0].new_columns_covered == ["rotor_temp", "wind_speed"]
    assert rounds[0].converged is False
    assert rounds[1].new_columns_covered == []
    assert rounds[1].converged is False  # only 1 quiet round so far
    assert rounds[2].converged is True

    # Blind check reuses the same fake_round under a fresh dataset_id, so it
    # should reach an identical brief - full agreement.
    assert result["blind_check"]["label_status_match"] is True
    assert result["blind_check"]["feature_overlap"] == 1.0


async def test_round_based_understanding_stops_at_max_rounds_when_never_converging(db_session, dataset_id, monkeypatch):
    embedding_client = GeminiEmbeddingClient()
    _stub_schema(monkeypatch, columns=[{"name": "a", "type": "text"}], row_count=10)

    async def fake_round(llm_client, embedding_client_, db_session_, table_name, ds_id, project_brief, round_number, query_log, blind_findings=None):
        return None  # never saves a brief - stays incomplete forever

    monkeypatch.setattr(rbu, "_run_one_round", fake_round)

    result = await rbu.run_round_based_understanding(
        llm_client=None,
        embedding_client=embedding_client,
        db_session=db_session,
        table_name="whatever",
        dataset_id=dataset_id,
        project_brief="test brief",
        max_rounds=3
    )

    assert result["converged"] is False
    assert result["rounds_run"] == 3
    assert result["blind_cycles"] == 0  # never even attempted a blind check

    round_store = UnderstandingRoundStore(db_session)
    rounds = round_store.get_rounds(dataset_id)
    assert len(rounds) == 3
    assert all(r.converged is False for r in rounds)

    # Neither brief ever got created, so the blind check has nothing to compare
    assert result["blind_check"]["label_status_match"] is None
    assert result["blind_check"]["feature_overlap"] is None


async def test_round_based_understanding_reconciles_new_blind_findings(db_session, dataset_id, monkeypatch):
    embedding_client = GeminiEmbeddingClient()
    knowledge_store = KnowledgeStore(embedding_client, db_session)
    brief_store = ModelingBriefStore(db_session)

    _stub_schema(monkeypatch, columns=[{"name": "wind_speed", "type": "double precision"}])

    blind_dataset_id = f"{dataset_id}__blind_check"
    call_log = []

    def _brief(ds_id):
        brief_store.save_brief(
            dataset_id=ds_id, label_status="absent", label_column=None, label_notes="n",
            feature_set=[{"column": "wind_speed", "role": "feature", "reason": "r"}],
            preprocessing_rules=_full_preprocessing_rules(),
            validation_strategy="v", confidence=0.8,
            validation_anchor=_NO_ANCHOR
        )

    async def fake_round(llm_client, embedding_client_, db_session_, table_name, ds_id, project_brief, round_number, query_log, blind_findings=None):
        call_log.append((ds_id, round_number, blind_findings))
        query_log.append('SELECT COUNT(*) FROM "t"')

        if ds_id == dataset_id and round_number == 1:
            # Main chain's real work: a brief that never mentions the thing
            # the blind check is about to independently find.
            _brief(ds_id)
            await knowledge_store.save_observation(ds_id, "wind_speed looks stable across the board.", 0.7)
        elif ds_id == blind_dataset_id:
            # Every blind pass independently reaches the same brief and the
            # same extra finding - idempotent thanks to upsert/dedup, so it
            # naturally agrees with itself (and, after reconciliation, with
            # main) on the second pass.
            _brief(ds_id)
            await knowledge_store.save_observation(
                ds_id, "Sensor XYZ shows a severe anomaly starting in March.", 0.95
            )
        # else: subsequent main-chain rounds (round_number > 1, any cycle) - no-op

    monkeypatch.setattr(rbu, "_run_one_round", fake_round)

    result = await rbu.run_round_based_understanding(
        llm_client=None,
        embedding_client=embedding_client,
        db_session=db_session,
        table_name="turbine_data",
        dataset_id=dataset_id,
        project_brief="test brief",
        max_rounds=10
    )

    assert result["converged"] is True
    assert result["blind_cycles"] == 2  # cycle 1 finds something new, cycle 2 agrees
    assert result["rounds_run"] == 5  # 3 rounds to first convergence + 2 more after reconciliation

    # The blind check's finding must have been merged into the MAIN dataset,
    # not left stranded under the blind namespace.
    main_contents = [o.content for o in knowledge_store.get_observations(dataset_id)]
    assert any("Sensor XYZ" in c for c in main_contents)

    # The round immediately after reconciliation must have been told about
    # it explicitly, not left to retrieval-similarity luck.
    reconciliation_calls = [c for c in call_log if c[0] == dataset_id and c[2] is not None]
    assert len(reconciliation_calls) == 1
    assert any("Sensor XYZ" in f for f in reconciliation_calls[0][2])


async def test_round_based_understanding_stops_at_max_blind_cycles(db_session, dataset_id, monkeypatch):
    embedding_client = GeminiEmbeddingClient()
    brief_store = ModelingBriefStore(db_session)

    _stub_schema(monkeypatch, columns=[{"name": "wind_speed", "type": "double precision"}])

    blind_dataset_id = f"{dataset_id}__blind_check"
    knowledge_store = KnowledgeStore(embedding_client, db_session)
    blind_call_count = {"n": 0}

    def _brief(ds_id):
        brief_store.save_brief(
            dataset_id=ds_id, label_status="absent", label_column=None, label_notes="n",
            feature_set=[{"column": "wind_speed", "role": "feature", "reason": "r"}],
            preprocessing_rules=_full_preprocessing_rules(),
            validation_strategy="v", confidence=0.8,
            validation_anchor=_NO_ANCHOR
        )

    async def fake_round(llm_client, embedding_client_, db_session_, table_name, ds_id, project_brief, round_number, query_log, blind_findings=None):
        query_log.append('SELECT COUNT(*) FROM "t"')
        if ds_id == dataset_id and round_number == 1:
            _brief(ds_id)
        elif ds_id == blind_dataset_id:
            _brief(ds_id)
            blind_call_count["n"] += 1
            await knowledge_store.save_observation(
                ds_id, f"Finding from blind pass {blind_call_count['n']}.", 0.9
            )

    # Force every blind-check diff to look non-empty regardless of actual
    # embedding distances - this test is about the max_blind_cycles safety
    # valve halting an endless reconciliation cycle, not dedup accuracy
    # (already covered elsewhere), so make "always something new" deterministic.
    async def never_duplicate(*args, **kwargs):
        return None

    monkeypatch.setattr(rbu, "_run_one_round", fake_round)
    monkeypatch.setattr(KnowledgeStore, "find_duplicate", never_duplicate)

    result = await rbu.run_round_based_understanding(
        llm_client=None,
        embedding_client=embedding_client,
        db_session=db_session,
        table_name="turbine_data",
        dataset_id=dataset_id,
        project_brief="test brief",
        max_rounds=10,
        max_blind_cycles=3
    )

    assert result["converged"] is False
    assert result["blind_cycles"] == 3
    assert blind_call_count["n"] == 3
