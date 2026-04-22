from unittest.mock import MagicMock


def test_link_graph_is_noop_and_skips_memgraph_writes():
    conn = MagicMock()
    driver = MagicMock()

    from rag.graph_linking import link_graph
    link_graph(conn, driver, "source-uuid", "job-uuid")

    conn.execute.assert_not_called()
    driver.session.assert_not_called()
