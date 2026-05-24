"""Tests for geometry helpers beyond projection math."""

from __future__ import annotations

import numpy as np
import pytest

from visionforge_loader.geometry import instance_id_to_class_id


def _make_meta(objects: list[dict]) -> dict:
    return {"objects": objects}


def test_instance_id_to_class_id_basic():
    """Two known instance ids + background (0) + unknown id map correctly."""
    meta = _make_meta([
        {"instance_id": 1, "class_id": 3},
        {"instance_id": 2, "class_id": 7},
    ])
    instance_id = np.array([[0.0, 1.0], [2.0, 99.0]], dtype=np.float32)

    out = instance_id_to_class_id(instance_id, meta)

    assert out.shape == (2, 2)
    assert out.dtype == np.int32
    assert out[0, 0] == 0   # background (id == 0)
    assert out[0, 1] == 3   # instance 1 → class 3
    assert out[1, 0] == 7   # instance 2 → class 7
    assert out[1, 1] == 0   # unknown id 99 → 0


def test_instance_id_to_class_id_empty_objects():
    """No objects in meta → all pixels map to 0."""
    meta = _make_meta([])
    instance_id = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    out = instance_id_to_class_id(instance_id, meta)

    assert out.shape == (1, 3)
    assert out.dtype == np.int32
    np.testing.assert_array_equal(out, 0)


def test_instance_id_to_class_id_missing_objects_key():
    """meta_json_raw without 'objects' key → all pixels map to 0."""
    out = instance_id_to_class_id(np.ones((4, 4), dtype=np.float32), {})

    assert out.shape == (4, 4)
    assert out.dtype == np.int32
    np.testing.assert_array_equal(out, 0)


def test_instance_id_to_class_id_float_ids_rounded():
    """Float instance ids close to an integer are rounded before lookup."""
    meta = _make_meta([{"instance_id": 5, "class_id": 2}])
    instance_id = np.array([[4.9, 5.0, 5.1]], dtype=np.float32)

    out = instance_id_to_class_id(instance_id, meta)

    assert out[0, 0] == 2   # 4.9 rounds to 5
    assert out[0, 1] == 2   # 5.0 → 5
    assert out[0, 2] == 2   # 5.1 rounds to 5
