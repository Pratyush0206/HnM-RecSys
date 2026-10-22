import numpy as np
import pandas as pd

from hmrec import metadata


def write_customers(directory):
    pd.DataFrame(
        {
            "customer_id": ["a", "b", "c", "z"],
            "FN": [1.0, np.nan, np.nan, 1.0],
            "Active": [1.0, np.nan, np.nan, 1.0],
            "club_member_status": ["ACTIVE", "PRE-CREATE", None, "LEFT CLUB"],
            "fashion_news_frequency": ["NONE", "Regularly", "Monthly", None],
            "age": [25.0, np.nan, 40.0, 70.0],
            "postal_code": ["p1", "p2", "p3", "p4"],
        }
    ).to_csv(directory / "customers.csv", index=False)


def write_articles(directory):
    pd.DataFrame(
        {
            "article_id": [10, 20, 30],
            "product_type_no": [253, 306, 253],
            "product_group_name": ["Garment Upper body", "Underwear", "Garment Upper body"],
            "department_no": [1, 2, 1],
            "colour_group_code": [9, 10, 9],
            "index_group_no": [1, 1, 1],
            "garment_group_no": [1001, 1002, 1001],
        }
    ).to_csv(directory / "articles.csv", index=False)


def test_customers_follow_the_customer_map_and_encode_categories(tmp_path):
    write_customers(tmp_path)
    customer_map = pd.DataFrame({"customer_id": ["a", "b", "c"], "customer_idx": [2, 0, 1]})
    result = metadata.build_customers(tmp_path, customer_map)
    assert result["customer_idx"].tolist() == [0, 1, 2]
    assert result["c_age"].isna().tolist() == [True, False, False]
    assert result["c_age"].iloc[1] == 40
    assert result["c_club"].fillna(-1).tolist() == [1.0, -1.0, 0.0]
    assert result["c_news"].tolist() == [1.0, 2.0, 0.0]


def test_articles_keep_only_items_in_the_data(tmp_path):
    write_articles(tmp_path)
    result = metadata.build_articles(tmp_path, np.array([10, 30, 999]))
    assert result["article_id"].tolist() == [10, 30]


def test_same_product_group_gets_the_same_code(tmp_path):
    write_articles(tmp_path)
    result = metadata.build_articles(tmp_path, np.array([10, 20, 30]))
    codes = result.set_index("article_id")["i_group"]
    assert codes[10] == codes[30]
    assert codes[10] != codes[20]


def test_group_codes_do_not_depend_on_which_items_are_present(tmp_path):
    write_articles(tmp_path)
    everything = metadata.build_articles(tmp_path, np.array([10, 20, 30]))
    only_one = metadata.build_articles(tmp_path, np.array([20]))
    full_code = everything.set_index("article_id")["i_group"][20]
    assert only_one.set_index("article_id")["i_group"][20] == full_code