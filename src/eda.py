import pandas as pd
from sklearn import preprocessing
import numpy as np
from eda_tools import *
PATH = "D:/Documents/Machine Learning/Walmart Trip Type"
DATA_IMPORT_PATH = PATH + "/data/"

CAT_DUMMIES = True
DAY_DUMMIES = True
DAY_SIN_COS = True
DPT_ENCODE = True
FAVOURITE_FINELINES =True
fav_fnls = 4
PERC_NAN = 0
export = True

if __name__ == '__main__':

    train, test, sample_submission = read_data()

    # enumareate days
    train = enum_days(train)
    test = enum_days(test)

    # and calculate day index
    train = day_index(train)
    test = day_index(test)

    # calculate week number
    train, test = week_number(train, test)
    test_correct = no_dup_upcs(test)
    train_correct = no_dup_upcs(train)

    #  visitnumber scancount sum, total items purchased
    train_new = train.groupby("VisitNumber")["ScanCount"]\
                     .agg("sum")\
                     .reset_index()\
                     .rename(columns={"ScanCount": "CartNetItems"})
    test_new = test.groupby("VisitNumber")["ScanCount"]\
                   .agg("sum")\
                   .reset_index()\
                   .rename(columns={"ScanCount": "CartNetItems"})
    train_new = items_bought(train, train_new)
    test_new = items_bought(test, test_new)

    # add triptype to each visitnumber
    train_new = add_triptype(train, train_new)

    # rearrange columns
    train_new = train_new[["VisitNumber", "TripType", "CartNetItems", "ItemsBought"]]

    # fineline entropy
    train_new = fineline_entropy(train, train_new)
    test_new = fineline_entropy(test, test_new)
    #fineline chaos
    train_new = fineline_chaos(train, train_new)
    test_new = fineline_chaos(test, test_new)

    train_new = refunded_exact_items(train_correct, train_new)
    test_new = refunded_exact_items(test_correct, test_new)
    train_new = replaced_refunded_not_exact(train_correct, train_new)
    test_new = replaced_refunded_not_exact(test_correct, test_new)

    # bought exactly the number of items i returned(same upc)
    train_new = replaced_exact_items(train_correct, train_new)
    test_new = replaced_exact_items(test_correct, test_new)

    # total items returned
    train_new = items_returned(train, train_new)
    test_new = items_returned(test, test_new)

    # returned to bought ratio
    train_new["RtnTobtRatio"] = (train_new["ItemsReturned"] / train_new["ItemsBought"]).round(3)
    test_new["RtnTobtRatio"] = (test_new["ItemsReturned"] / test_new["ItemsBought"]).round(3)

    # Total Items
    train_new["TotalItems"] = train_new["ItemsBought"] + train_new["ItemsReturned"]
    test_new["TotalItems"] = test_new["ItemsBought"] + test_new["ItemsReturned"]

    # add day eum
    temp = train.groupby(["VisitNumber", "WeekdayEnum"]).first().reset_index()
    train_new = train_new.merge(temp[["WeekdayEnum", "VisitNumber"]], on="VisitNumber", how="left")
    temp = test.groupby(["VisitNumber", "WeekdayEnum"]).first().reset_index()
    test_new = test_new.merge(temp[["WeekdayEnum", "VisitNumber"]], on="VisitNumber", how="left")

    train["DepartmentDescription"] = train["DepartmentDescription"].fillna("None")
    test["DepartmentDescription"] = test["DepartmentDescription"].fillna("None")

    # nunique UPCs for buyght items and returned
    temp = train[train["ScanCount"] > 0].groupby("VisitNumber")["Upc"].nunique().reset_index()
    train_new = train_new.merge(temp, how="left", on="VisitNumber").rename(columns={"Upc": "UniqueUpcsBought"})
    train_new["UniqueUpcsBought"] = train_new["UniqueUpcsBought"].fillna(0)

    temp = test[test["ScanCount"] > 0].groupby("VisitNumber")["Upc"].nunique().reset_index()
    test_new = test_new.merge(temp, how="left", on="VisitNumber").rename(columns={"Upc": "UniqueUpcsBought"})
    test_new["UniqueUpcsBought"] = test_new["UniqueUpcsBought"].fillna(0)

    temp = train[train["ScanCount"] < 0].groupby("VisitNumber")["Upc"].nunique().reset_index()
    train_new = train_new.merge(temp, how="left", on="VisitNumber").rename(columns={"Upc": "UniqueUpcsReturned"})
    train_new["UniqueUpcsReturned"] = train_new["UniqueUpcsReturned"].fillna(0)

    temp = test[test["ScanCount"] < 0].groupby("VisitNumber")["Upc"].nunique().reset_index()
    test_new = test_new.merge(temp, how="left", on="VisitNumber").rename(columns={"Upc": "UniqueUpcsReturned"})
    test_new["UniqueUpcsReturned"] = test_new["UniqueUpcsReturned"].fillna(0)

    # total unique UPCs
    train_new["TotalUniqueUpcs"] = train_new["UniqueUpcsBought"] + train_new["UniqueUpcsReturned"]
    test_new["TotalUniqueUpcs"] = test_new["UniqueUpcsBought"] + test_new["UniqueUpcsReturned"]

    # total unique UPCs to total items perc
    train_new["TotalUniqueUpcs_perc"] = (train_new["TotalUniqueUpcs"] / train_new["TotalItems"]).round(3)
    test_new["TotalUniqueUpcs_perc"] = (test_new["TotalUniqueUpcs"] / test_new["TotalItems"]).round(3)

    # UniqueUpcs bought /items bought
    train_new["UniqueUpcsBought_perc"] = (train_new["UniqueUpcsBought"] / train_new["ItemsBought"]).round(3)
    train_new["UniqueUpcsReturned_perc"] = (train_new["UniqueUpcsReturned"] / train_new["ItemsReturned"]).round(3)

    train_new["UniqueUpcsBought_perc"] = train_new["UniqueUpcsBought_perc"].fillna(PERC_NAN)
    train_new["UniqueUpcsReturned_perc"] = train_new["UniqueUpcsReturned_perc"].fillna(PERC_NAN)
    test_new["UniqueUpcsBought_perc"] = (test_new["UniqueUpcsBought"] / test_new["ItemsBought"]).round(3)
    test_new["UniqueUpcsReturned_perc"] = (test_new["UniqueUpcsReturned"] / test_new["ItemsReturned"]).round(3)

    test_new["UniqueUpcsBought_perc"] = test_new["UniqueUpcsBought_perc"].fillna(PERC_NAN)
    test_new["UniqueUpcsReturned_perc"] = test_new["UniqueUpcsReturned_perc"].fillna(PERC_NAN)

    # unique finelineNumbers
    temp = train[train["ScanCount"] > 0].groupby("VisitNumber")["FinelineNumber"].nunique().reset_index()
    train_new = train_new.merge(temp, how="left", on="VisitNumber")\
                         .rename(columns={"FinelineNumber": "UniqueFinelinesBought"})
    train_new["UniqueFinelinesBought"] = train_new["UniqueFinelinesBought"].fillna(0)

    temp = test[test["ScanCount"] > 0].groupby("VisitNumber")["FinelineNumber"].nunique().reset_index()
    test_new = test_new.merge(temp, how="left", on="VisitNumber")\
                       .rename(columns={"FinelineNumber": "UniqueFinelinesBought"})

    test_new["UniqueFinelinesBought"] = test_new["UniqueFinelinesBought"].fillna(0)

    temp = train[train["ScanCount"] < 0].groupby("VisitNumber")["FinelineNumber"]\
                                        .nunique()\
                                        .reset_index()
    train_new = train_new.merge(temp, how="left", on="VisitNumber")\
                         .rename(columns={"FinelineNumber": "UniqueFinelinesReturned"})

    train_new["UniqueFinelinesReturned"] = train_new["UniqueFinelinesReturned"].fillna(0)
    temp = test[test["ScanCount"] < 0].groupby("VisitNumber")["FinelineNumber"]\
                                      .nunique()\
                                      .reset_index()
    test_new = test_new.merge(temp, how="left", on="VisitNumber")\
                       .rename(columns={"FinelineNumber": "UniqueFinelinesReturned"})
    test_new["UniqueFinelinesReturned"] = test_new["UniqueFinelinesReturned"].fillna(0)

    # total Unique finelines
    train_new["TotalUniqueFinelines"] = train_new["UniqueFinelinesReturned"] + train_new["UniqueFinelinesBought"]
    test_new["TotalUniqueFinelines"] = test_new["UniqueFinelinesReturned"] + test_new["UniqueFinelinesBought"]

    # total unique finlines to total items percentage
    train_new["TotalUniqueFinelines_perc"] = (train_new["TotalUniqueFinelines"] / train_new["TotalItems"]).round(3)
    test_new["TotalUniqueFinelines_perc"] = (test_new["TotalUniqueFinelines"] / test_new["TotalItems"]).round(3)

    # finelines percentage
    train_new["UniqueFinelinesBought_perc"] = (train_new["UniqueFinelinesBought"] / train_new["ItemsBought"]).round(3)
    train_new["UniqueFinelinesReturned_perc"] = (train_new["UniqueFinelinesReturned"] / train_new["ItemsReturned"])\
                                                .round(3)

    train_new["UniqueFinelinesBought_perc"] = train_new["UniqueFinelinesBought_perc"].fillna(PERC_NAN)
    train_new["UniqueFinelinesReturned_perc"] = train_new["UniqueFinelinesReturned_perc"].fillna(PERC_NAN)
    test_new["UniqueFinelinesBought_perc"] = (test_new["UniqueFinelinesBought"] / test_new["ItemsBought"]).round(3)
    test_new["UniqueFinelinesReturned_perc"] = (test_new["UniqueFinelinesReturned"] / test_new["ItemsReturned"]).round(
        3)

    test_new["UniqueFinelinesBought_perc"] = test_new["UniqueFinelinesBought_perc"].fillna(PERC_NAN)
    test_new["UniqueFinelinesReturned_perc"] = test_new["UniqueFinelinesReturned_perc"].fillna(PERC_NAN)

    # finelines entropy

    train_new.replace(np.inf, 0, inplace=True)
    test_new.replace(np.inf, 0, inplace=True)

    if FAVOURITE_FINELINES:
        train_new = get_fav_flnumbers(train, train_new, n=fav_fnls)
        test_new = get_fav_flnumbers(test, test_new, n=fav_fnls)

    if CAT_DUMMIES:
        train_new = category_dummies(train, train_new, train=True)
        test_new = category_dummies(test, test_new)
        train_new = train_new.drop(columns="HEALTH AND BEAUTY AIDS", axis=1)

    # onehot encode days
    if DAY_DUMMIES:
        train_new = day_dummies(train, train_new, train=True)
        test_new = day_dummies(test, test_new)

    if DAY_SIN_COS:
        train_new = days_sin_cos(train_new)
        test_new = days_sin_cos(test_new)
        print()
    # encode department description
    if DPT_ENCODE:
        department_description_le = preprocessing.LabelEncoder()
        department_description_le.fit(pd.concat([train['DepartmentDescription'], test["DepartmentDescription"]]))
        department_description_mapping = dict(zip(department_description_le.classes_,
                                                  department_description_le.transform(department_description_le.classes_)))
        department_description_mapping_inv = dict(
            zip(department_description_le.transform(department_description_le.classes_),
                department_description_le.classes_))
        train["DepartmentDescription"] = train["DepartmentDescription"].map(department_description_mapping)

    if export:
        chonda = ""
        if DAY_SIN_COS:
            chonda = chonda + "_sincos"
        if CAT_DUMMIES:
            chonda = chonda + "_catdum"
        if DAY_DUMMIES:
            chonda = chonda + "_daydum"
        if FAVOURITE_FINELINES:
            chonda = chonda + "_" + str(fav_fnls) + "favFL"
        train_new.to_csv(PATH + "/eda final/" + "train_new_temp" + chonda+".csv", index=False)
        # train.to_csv(PATH + "/temp/" + "train_temp.csv", index=False)
        # test.to_csv(PATH + "/temp/" + "test_temp.csv", index=False)
        test_new.to_csv(PATH + "/eda final/" + "test_new_temp" + chonda + ".csv", index=False)

    # return train_new, test_new
