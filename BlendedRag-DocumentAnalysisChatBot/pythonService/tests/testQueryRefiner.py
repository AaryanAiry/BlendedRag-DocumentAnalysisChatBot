def test_refiner_basic():
    from app.rag.queryRefiner import refine_query_intelligent
    out = refine_query_intelligent("Give me a summary of system architecture and pricing details.")
    assert "refinedQuery" in out and out["refinedQuery"]
    assert isinstance(out["subQueries"], list)
    assert isinstance(out["keywords"], list)
    assert isinstance(out["variants"], list)
