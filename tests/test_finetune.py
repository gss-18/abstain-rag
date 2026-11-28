import torch

from abstain_rag.finetune import batches, contrastive_loss, training_passages


def test_training_passages_skip_held_out_sections_and_short_text():
    long = " ".join(["word"] * 80)
    passages = [{"_id": "1.1#0", "section": "1.1", "text": long},
                {"_id": "1.1#1", "section": "1.1", "text": long},
                {"_id": "2.1#0", "section": "2.1", "text": long},
                {"_id": "3.1#0", "section": "3.1", "text": "too short"}]
    out = training_passages(passages, held_out_sections={"2.1"}, per_section=1)
    assert [p["section"] for p in out] == ["1.1"]


def test_batches_never_repeat_a_section_and_keep_every_pair():
    pairs = [(f"q{i}", f"p{i}", f"s{i % 3}") for i in range(9)]
    out = batches(pairs, batch_size=4, seed=0)
    assert sorted(p for b in out for p in b) == sorted(pairs)
    for b in out:
        assert len({p[2] for p in b}) == len(b)


def test_contrastive_loss_is_low_when_each_query_matches_its_passage():
    eye = torch.eye(4)
    shuffled = eye[[1, 2, 3, 0]]
    assert contrastive_loss(eye, eye) < contrastive_loss(eye, shuffled)
