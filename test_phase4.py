from server.tasks import (
    EpisodeRecord, RubricComposer, Dimension, Gate,
    grade_task1, grade_task2, grade_task3,
    grade_episode, grade_episode_detailed,
)
from server.farming_environment import FarmingEnvironment
from models import FarmAction


def test_grade_task1_perfect():
    # net_worth = 400 (exactly 2x target), full stewardship, no wither
    record = EpisodeRecord(
        task_id=1, initial_money=200.0, final_money=350.0, storage_value=50.0,
        total_reward=10.0, days_elapsed=30, max_days=30,
        withered_count=0, drought_days=0, healthy_days=30, sell_events=[]
    )
    score = grade_task1(record)
    assert score >= 0.99, f"expected near-max score, got {score}"
    print(f"grade_task1 perfect: {score}")


def test_grade_task1_bankrupt():
    record = EpisodeRecord(
        task_id=1, initial_money=200.0, final_money=0.0, storage_value=0.0,
        total_reward=-5.0, days_elapsed=10, max_days=30,
        withered_count=5, drought_days=0, healthy_days=0, sell_events=[]
    )
    score = grade_task1(record)
    assert score <= 0.01, f"expected floor score for bankrupt, got {score}"
    print(f"grade_task1 bankrupt: {score}")


def test_grade_task2_timing_rewarded():
    sell_events = [
        {"day": 10, "crop": "wheat", "qty": 10.0, "price": 10.0, "base_price": 8.0},
        {"day": 20, "crop": "corn",  "qty": 20.0, "price": 25.0, "base_price": 20.0},
    ]
    record_good = EpisodeRecord(
        task_id=2, initial_money=150.0, final_money=300.0, storage_value=75.0,
        total_reward=15.0, days_elapsed=45, max_days=45,
        withered_count=0, drought_days=0, healthy_days=30,
        sell_events=sell_events
    )
    sell_events_bad = [
        {"day": 10, "crop": "wheat", "qty": 10.0, "price": 5.0, "base_price": 8.0},
    ]
    record_bad = EpisodeRecord(
        task_id=2, initial_money=150.0, final_money=300.0, storage_value=75.0,
        total_reward=15.0, days_elapsed=45, max_days=45,
        withered_count=0, drought_days=0, healthy_days=30,
        sell_events=sell_events_bad
    )
    good_score = grade_task2(record_good)
    bad_score  = grade_task2(record_bad)
    assert good_score > bad_score, f"good timing should score higher: {good_score} vs {bad_score}"
    print(f"grade_task2 timing: good={good_score} bad={bad_score}")


def test_grade_task3_survival_matters():
    # survived: net_worth=110 >= initial (solvency gate passes), full episode
    record_survived = EpisodeRecord(
        task_id=3, initial_money=100.0, final_money=80.0, storage_value=30.0,
        total_reward=5.0, days_elapsed=60, max_days=60,
        withered_count=1, drought_days=60, healthy_days=30, sell_events=[]
    )
    record_bankrupt = EpisodeRecord(
        task_id=3, initial_money=100.0, final_money=0.0, storage_value=0.0,
        total_reward=-10.0, days_elapsed=20, max_days=60,
        withered_count=5, drought_days=20, healthy_days=5, sell_events=[]
    )
    score_survived = grade_task3(record_survived)
    score_bankrupt = grade_task3(record_bankrupt)
    assert score_survived > score_bankrupt
    print(f"grade_task3 survival: survived={score_survived} bankrupt={score_bankrupt}")


def test_rubric_composer_detail():
    record = EpisodeRecord(
        task_id=1, initial_money=200.0, final_money=300.0, storage_value=100.0,
        total_reward=8.0, days_elapsed=30, max_days=30,
        withered_count=0, drought_days=0, healthy_days=20, sell_events=[]
    )
    result = grade_episode_detailed(record)
    assert "score"      in result
    assert "dimensions" in result
    assert "gated"      in result
    assert "profit"      in result["dimensions"]
    assert "stewardship" in result["dimensions"]
    assert "efficiency"  in result["dimensions"]
    assert 0.0 < result["score"] < 1.0
    print(f"rubric_detail: score={result['score']} dims={result['dimensions']}")


def test_all_scores_in_range():
    import random
    random.seed(7)
    for task_id in [1, 2, 3]:
        for _ in range(10):
            record = EpisodeRecord(
                task_id=task_id,
                initial_money=[200.0, 150.0, 100.0][task_id - 1],
                final_money=random.uniform(0, 500),
                storage_value=random.uniform(0, 100),
                total_reward=random.uniform(-20, 20),
                days_elapsed=random.randint(5, 60),
                max_days=[30, 45, 60][task_id - 1],
                withered_count=random.randint(0, 6),
                drought_days=random.randint(0, 60),
                healthy_days=random.randint(0, 60),
                sell_events=[],
            )
            score = grade_episode(record)
            assert 0.0 <= score <= 1.0, f"task {task_id} score out of range: {score}"
    print("all_scores_in_range() OK — 30 random records all scored in [0.0, 1.0]")


def test_end_to_end_grading():
    env = FarmingEnvironment(task_id=1)
    env.reset(seed=0)
    obs = None
    for _ in range(35):
        obs = env.step(FarmAction(action_type="wait"))
        if obs.done:
            break
    assert obs.done
    assert "grade"          in obs.metadata
    assert "grade_detailed" in obs.metadata
    assert 0.0 <= obs.metadata["grade"] <= 1.0
    detail = obs.metadata["grade_detailed"]
    assert "score"      in detail
    assert "dimensions" in detail
    print(f"end_to_end_grading() OK — grade={obs.metadata['grade']} dims={detail['dimensions']}")


if __name__ == "__main__":
    test_grade_task1_perfect()
    test_grade_task1_bankrupt()
    test_grade_task2_timing_rewarded()
    test_grade_task3_survival_matters()
    test_rubric_composer_detail()
    test_all_scores_in_range()
    test_end_to_end_grading()
    print("\nPhase 4 complete — all grader tests passed")
