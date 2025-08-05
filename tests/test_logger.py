import os
import importlib
import pytest
import tempfile


@pytest.fixture
def setup_logger(monkeypatch):
    # Create a named temporary file that persists
    with tempfile.NamedTemporaryFile(delete=False, mode='w+', encoding='utf-8') as temp_file:
        log_path = temp_file.name

    monkeypatch.setenv("LOG_FILE", log_path)
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    # Reload to reset logger state
    import utils.logger
    importlib.reload(utils.logger)

    logger = utils.logger.get_logger("test.logger")

    yield logger, log_path

    # Clean up handlers and file
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

    os.remove(log_path)


def test_logger_writes_info_to_file(setup_logger):
    logger, path = setup_logger
    message = "This is an info message"
    logger.info(message)

    for handler in logger.handlers:
        handler.flush()

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    assert message in content


def test_logger_respects_env_log_level(setup_logger):
    logger, path = setup_logger
    message = "This is a debug message"
    logger.debug(message)

    for handler in logger.handlers:
        handler.flush()

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    assert message in content
