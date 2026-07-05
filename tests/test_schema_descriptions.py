import inspect

import pytest

import rag.api.schemas as schemas
from pydantic import BaseModel

_REQUEST_MODELS = {
    schemas.DescribeImageRequest,
    schemas.SearchRequest,
    schemas.SearchOptions,
    schemas.RetrieveRequest,
    schemas.RetrieveOptions,
    schemas.AnswerRequest,
    schemas.CommunityOptions,
    schemas.CommunityRequest,
}


def _request_model_classes():
    return [m for _, m in inspect.getmembers(schemas, inspect.isclass)
            if issubclass(m, BaseModel) and m in _REQUEST_MODELS]


def _all_model_classes():
    return {m for _, m in inspect.getmembers(schemas, inspect.isclass)
            if issubclass(m, BaseModel) and m is not BaseModel}


def test_request_models_are_base_model_subclasses():
    assert _REQUEST_MODELS.issubset(_all_model_classes()), \
        f"Some _REQUEST_MODELS are not BaseModel subclasses: {_REQUEST_MODELS - _all_model_classes()}"


@pytest.mark.parametrize("model_cls", _request_model_classes())
def test_schema_descriptions(model_cls: type[BaseModel]):
    for field_name, field_info in model_cls.model_fields.items():
        desc = field_info.description
        assert desc is not None, f"{model_cls.__name__}.{field_name} is missing a description"
        assert len(desc.strip()) > 0, f"{model_cls.__name__}.{field_name} has an empty description"
