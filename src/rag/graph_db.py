from contextlib import contextmanager

from neo4j import GraphDatabase

from rag.config import settings


@contextmanager
def get_graph_driver():
    driver = GraphDatabase.driver(settings.MEMGRAPH_URL, auth=None)
    try:
        yield driver
    finally:
        driver.close()
