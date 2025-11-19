from unittest.mock import MagicMock, patch

from app.services.vdb.pinecone_client import get_pinecone_client, get_pinecone_index


@patch("app.services.vdb.pinecone_client.Pinecone")
def test_pinecone_client_init(mock_pc):
    # Reset global state to avoid caching issues
    import app.services.vdb.pinecone_client as pc_module

    pc_module._pc = None

    mock_instance = MagicMock()
    mock_pc.return_value = mock_instance

    client = get_pinecone_client()
    assert client == mock_instance  # nosec


@patch("app.services.vdb.pinecone_client.Pinecone")
def test_pinecone_get_index(mock_pc):
    # Reset global state to avoid caching issues
    import app.services.vdb.pinecone_client as pc_module

    pc_module._index = None
    pc_module._pc = None

    mock_client = MagicMock()
    mock_pc.return_value = mock_client

    # simulate list_indexes()
    mock_client.list_indexes.return_value = []

    mock_index = MagicMock()
    mock_client.Index.return_value = mock_index

    index = get_pinecone_index()

    assert index == mock_index  # nosec
    mock_client.create_index.assert_called_once()
