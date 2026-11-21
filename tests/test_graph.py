"""Tests for the 64-hexagram state graph (Phase 0).

Conventions under test (see README):
- yang = 1, yin = 0; bits ordered bottom-to-top, L1..L6.
- Node identity is the 6-bit string, e.g. "111111" (Qian).
- Edge types: bian (flip one line, Q6), cuo (bitwise NOT),
  zong (reverse bit order), hu (nuclear, directed).
"""

import pytest

from taiji_field.data import CHINESE_NAMES, load_hexagrams
from taiji_field.graph import (
    build_graph,
    build_nuclear_graph,
    cuo_bits,
    hu_bits,
    zong_bits,
)


@pytest.fixture(scope="module")
def hexagrams():
    return load_hexagrams()


@pytest.fixture(scope="module")
def graph():
    return build_graph()


@pytest.fixture(scope="module")
def nuclear():
    return build_nuclear_graph()


def _edges_of_type(graph, kind):
    return [(u, v) for u, v, k in graph.edges(keys=True) if k == kind]


# --- data integrity -------------------------------------------------------

def test_loads_64_hexagrams(hexagrams):
    assert len(hexagrams) == 64


def test_bits_are_unique_and_well_formed(hexagrams):
    bits = [h.bits for h in hexagrams]
    assert len(set(bits)) == 64
    assert all(len(b) == 6 and set(b) <= {"0", "1"} for b in bits)


def test_king_wen_numbers_cover_1_to_64(hexagrams):
    assert sorted(h.king_wen for h in hexagrams) == list(range(1, 65))


def test_chinese_names_complete_and_unique():
    assert len(CHINESE_NAMES) == 64
    assert set(CHINESE_NAMES) == set(range(1, 65))
    assert len(set(CHINESE_NAMES.values())) == 64
    assert all(isinstance(n, str) and n for n in CHINESE_NAMES.values())


def test_chinese_names_spot_check(hexagrams):
    by_kw = {h.king_wen: h for h in hexagrams}
    assert CHINESE_NAMES[1] == "乾" and by_kw[1].pinyin == "Qian"
    assert CHINESE_NAMES[2] == "坤" and by_kw[2].pinyin == "Kun"
    assert CHINESE_NAMES[24] == "复" and by_kw[24].pinyin == "Fu"
    assert CHINESE_NAMES[63] == "既济" and by_kw[63].pinyin == "Ji Ji"
    assert CHINESE_NAMES[64] == "未济" and by_kw[64].pinyin == "Wei Ji"


def test_spot_check_known_hexagrams(hexagrams):
    by_kw = {h.king_wen: h for h in hexagrams}
    assert by_kw[1].bits == "111111"  # Qian 乾
    assert by_kw[2].bits == "000000"  # Kun 坤
    assert by_kw[24].bits == "100000"  # Fu ䷗ 复: single yang at L1 (bottom)
    assert by_kw[64].bits == "010101"  # Wei Ji ䷿ 未济


def test_find_hexagram_by_bits():
    from taiji_field.graph import find_hexagram

    # A 6-digit 0/1 string is bits, not a King Wen number (regression:
    # "100000".isdigit() must not swallow it as number 100000).
    found = find_hexagram("100000")
    assert found is not None and found.king_wen == 24
    assert find_hexagram("24").king_wen == 24
    assert find_hexagram("Fu").king_wen == 24
    assert find_hexagram("䷗").king_wen == 24
    assert find_hexagram("nonsense") is None
    assert find_hexagram("99") is None


# --- graph structure ------------------------------------------------------

def test_graph_has_64_nodes(graph):
    assert graph.number_of_nodes() == 64


def test_node_attributes(graph):
    for node, attrs in graph.nodes(data=True):
        assert attrs["king_wen"] in range(1, 65)
        assert attrs["name_zh"]
        assert attrs["pinyin"]
        assert attrs["bits"] == node


def test_bian_is_exactly_Q6(graph):
    bian = _edges_of_type(graph, "bian")
    assert len(bian) == 192  # 64 * 6 / 2
    for node in graph.nodes:
        nbrs = [v for u, v in bian if u == node] + [u for u, v in bian if v == node]
        assert len(nbrs) == 6, node
    for u, v in bian:
        assert sum(a != b for a, b in zip(u, v)) == 1  # Hamming distance 1


def test_bian_edge_records_which_line_flips(graph):
    for u, v, k, attrs in graph.edges(keys=True, data=True):
        if k != "bian":
            continue
        line = attrs["line"]
        assert 1 <= line <= 6
        flipped = u[: line - 1] + ("1" if u[line - 1] == "0" else "0") + u[line:]
        assert flipped == v


def test_cuo_is_exactly_32_pairs(graph, hexagrams):
    cuo = _edges_of_type(graph, "cuo")
    assert len(cuo) == 32
    for node in graph.nodes:
        nbrs = [v for u, v in cuo if u == node] + [u for u, v in cuo if v == node]
        assert len(nbrs) == 1, node
        assert nbrs[0] == cuo_bits(node)
    # cross-check against the dataset's opposite_bits for all 64
    by_bits = {h.bits: h for h in hexagrams}
    for h in hexagrams:
        assert cuo_bits(h.bits) == by_bits[h.bits].opposite_bits


def test_cuo_spot_check_qian_kun():
    assert cuo_bits("111111") == "000000"  # Qian <-> Kun
    assert cuo_bits("000000") == "111111"


def test_zong_28_pairs_plus_8_self_symmetric(graph, hexagrams):
    # cross-check against the dataset's inverted_bits for all 64
    by_bits = {h.bits: h for h in hexagrams}
    for h in hexagrams:
        assert zong_bits(h.bits) == by_bits[h.bits].inverted_bits
    zong = _edges_of_type(graph, "zong")
    pairs = [(u, v) for u, v in zong if u != v]
    loops = [(u, v) for u, v in zong if u == v]
    assert len(pairs) == 28
    assert len(loops) == 8
    self_symmetric_kw = sorted(
        graph.nodes[n]["king_wen"] for n, _ in loops
    )
    assert self_symmetric_kw == [1, 2, 27, 28, 29, 30, 61, 62]


def test_hu_image_is_exactly_16(nuclear, hexagrams):
    assert nuclear.number_of_nodes() == 64
    assert nuclear.number_of_edges() == 64  # one outgoing edge per hexagram
    image = {v for _, v in nuclear.edges()}
    assert len(image) == 16
    for target in image:
        preimages = [u for u, v in nuclear.edges() if v == target]
        assert len(preimages) == 4, target
    # cross-check against the dataset's nuclear_bits for all 64
    by_bits = {h.bits: h for h in hexagrams}
    for h in hexagrams:
        assert hu_bits(h.bits) == by_bits[h.bits].nuclear_bits
    assert hu_bits("111111") == "111111"  # nuclear of Qian is Qian


def test_hu_is_directed(nuclear):
    # Fu (100000: one yang at L1) -> Kun (000000) is one-way: Kun's
    # nuclear is itself, so there is no edge back from Kun to Fu.
    assert hu_bits("100000") == "000000"
    assert nuclear.has_edge("100000", "000000")
    assert not nuclear.has_edge("000000", "100000")
    assert nuclear.out_degree("100000") == 1
