# Description

This is a takeout recommendation dataset which contains a vast amount of meta information from Meituan Takeout app. We collect orders from 11 commercial districts in Beijing between March 1st and March 28th, 2021. The first three weeks of orders are as training, while the last week is used for testing to avoid data leakage. We briefly summarize each file as follows and for more details, please refer to README.md.

1. users.txt (attributes of all users)
2. pois.txt (attributes of all takeout restaurants)
3. spus.txt (attributes of all food)
4. orders_poi_session.txt (a sequence of restaurants clicked by user before ordering)
5. orders_spu_train.txt (order-food in training set)
6. orders_train.txt (order-restaurant in training set)
7. orders_test.txt (order-restaurant in training set)
8. orders_poi_test_label.txt (test labels of order-restaurant)
9. orders_spu_test_label.txt (test labels of order-food)


# Detail
orders_train.txt
| Attribute         | Description                                |
|------------------------|-------------------------------------------------|
| wm_food_spu_id         | id of order food                                |
| user_id                | id of user                                      |
| wm_poi_id              | id of takeout restaurant                        |
| aor_id                 | address of  takeout restaurant                   
| order_price            | range of price                                  |
| order_timestamp        | timestamp of order                              |
| ord_period_name        | Time period of orfer (breakfast, lunch, dinner) |
| order_scene_name       | scene id of order                               |
| aoi_id                 | id of user's dilivery address                 |
| takedlvr_aoi_type_name | address type of aoi | 
| wm_food_spu_id         | id of order food                                |
| dt | date of order|

users.txt

| Attribute            | Description                            |
|----------------------|----------------------------------------|
| user_id              | id of user                             |
| avg_pay_amt          | historical average consumption         |
| avg_pay_amt_weekdays | weekly historical average consumption  |
| avg_pay_amt_weekends | weekend historical average consumption |


pois.txt
| Attribute                  | Description                    |
|----------------------------|--------------------------------|
| wm_poi_id                  | id of takeout restaurant       |
| wm_poi_name                | name of takeout restaurant     |
| primary_first_tag_id       | first level category of food   |
| primary_second_tag_id      | second level category of food  |
| primary_third_tag_id       | third level category of food   |
| poi_brand_id               | brand id of takeout restaurant |
| aor_id                     | address                        |
| poi_score                  | score of takeout restaurant    |
| delivery_comment_avg_score | score of delivery              |
| food_comment_avg_score     | score of food                  |

spus.txt
| Attribute        | Description           |
|------------------|-----------------------|
| wm_food_spu_id   | id of food            |
| wm_food_spu_name | name of food          |
| price            | price of food         |
| category         | category of food      |
| ingredients      | ingredients of food   |
| taste            | taste of food         |
| standfood_id     | standard id of food   |
| standfood_name   | standard name of food |

orders_poi_session.txt & orders_spu_train.tx

| Attribute      | Description                |
|----------------|----------------------------|
| wm_order_id    | id of order                |
| clicks         | a sequence of restaurants clicked by user before ordering                          |
| wm_food_spu_id | id of food that is ordered |

