import pandas as pd
from sklearn import preprocessing
import numpy as np
from scipy.stats import entropy
from math import log, e
PATH = "D:/Documents/Machine Learning/Walmart Trip Type"
DATA_IMPORT_PATH = PATH + "/data/"


def read_data(train_n_rows=None, test_n_rows=None):
    """
    Read train and test data
    """
    train = pd.read_csv(DATA_IMPORT_PATH + "train.csv", nrows=train_n_rows)
    test = pd.read_csv(DATA_IMPORT_PATH + "test.csv", nrows=test_n_rows)
    sample_submission = pd.read_csv(DATA_IMPORT_PATH + "sample_submission.csv")
    print("files read")
    return train, test, sample_submission


def print_groupby(grouped_df):
    """
    Print a GroupBy object for visualization perpose
    """
    for key, item in grouped_df:
        print(grouped_df.get_group(key), "\n\n")


def get_replaced_exact_items(upc_diff, scanCount_diff, scanCount_abs):
    """
    every row that has upc_diff=0 (same prodiuct) and scanCount_diff=0
    (same number of products returned and bought)
    """
    if (upc_diff == 0) & (scanCount_diff == 0.0):
        return scanCount_abs
    else:
        return 0


def get_replaced_notexact_items(scan_count_sum, scan_count_abs):
    """
    Return replaced Items
    """
    if scan_count_sum > 0:
        return scan_count_abs - scan_count_sum
    else:
        return 0


def get_refunded_notexact_items(scan_count_sum, scan_count_abs):
    """
    Return  refunded Items.
    """
    if scan_count_sum < 0:
        return abs(scan_count_sum)
    else:
        return 0


def delete_duplicates(replaced_items, scancount):
    if (replaced_items > 0) & (scancount < 0):
        return np.nan
    else:
        return replaced_items


def replaced_exact_items(correct_df, df_new):
    """
    Return  replaced exact items(den pira parapanw oute parakatw apo auta pou epestrepsa)
    """
    temp = correct_df.groupby(["VisitNumber"])\
                    .apply(pd.DataFrame.sort_values, ['Upc', "ScanCount"], ascending=True)\
                    .reset_index(drop=True)

    temp["Upc_diff"] = temp.groupby("VisitNumber")["Upc"].diff()
    temp["ScanCount_abs"] = temp["ScanCount"].apply(lambda x: abs(x))

    temp1 = temp.groupby(["VisitNumber", "Upc"])\
                .agg({"ScanCount": "sum"}).reset_index()\
                .rename(columns={"ScanCount": "ScanCount_sum"})
    temp = temp.merge(temp1, how="left", on=["VisitNumber", "Upc"])

    temp["ReplacedExactItems"] = temp.apply(lambda row: get_replaced_exact_items(row["Upc_diff"],
                                                                                 row["ScanCount_sum"],
                                                                                 row["ScanCount_abs"]), axis=1)
    temp = temp[temp["ReplacedExactItems"] > 0]
    temp = temp.groupby("VisitNumber")["ReplacedExactItems"].agg("sum").reset_index()
    df_new = df_new.merge(temp, how="left", on="VisitNumber")
    df_new = df_new.fillna(0)
    return df_new


def refunded_exact_items(correct_df, df_new):
    """
    Return  refunded Items.
    """
    correct_df = correct_df.groupby(["VisitNumber"])\
                           .apply(pd.DataFrame.sort_values, ['Upc', "ScanCount"], ascending=True)\
                           .reset_index(drop=True)

    correct_df["Upc_diff"] = correct_df.groupby("VisitNumber")["Upc"].diff()
    correct_df["ScanCount_abs"] = correct_df["ScanCount"].apply(lambda x: abs(x))
    temp1 = correct_df.groupby(["VisitNumber", "Upc"]).agg({"ScanCount": "sum"}).reset_index()
    temp1 = temp1.rename(columns={"ScanCount": "ScanCount_sum"})

    correct_df = correct_df.merge(temp1, how="left", on=["VisitNumber", "Upc"])
    correct_df = correct_df[(correct_df["Upc_diff"].isnull()) & (correct_df["ScanCount_sum"] < 0)]
    correct_df = correct_df.rename(columns={"ScanCount_sum": "RefundedExactItems"})
    correct_df["RefundedExactItems"] = correct_df["RefundedExactItems"].apply(lambda x: abs(x))
    correct_df = correct_df.groupby("VisitNumber")["RefundedExactItems"]\
                           .agg("sum")\
                           .reset_index()

    df_new = df_new.merge(correct_df, how="left", on="VisitNumber")
    df_new["RefundedExactItems"] = df_new["RefundedExactItems"].fillna(0)
    return df_new


def replaced_refunded_not_exact(correct_df, df_new):
    """
    Return  refunded and replaced items not exapct( bought or retured, one iss bigger thatn the other
    """
    correct_df = correct_df.groupby(["VisitNumber"])\
                           .apply(pd.DataFrame.sort_values, ['Upc', "ScanCount"], ascending=True)\
                           .reset_index(drop=True)

    correct_df["Upc_diff"] = correct_df.groupby("VisitNumber")["Upc"].diff()
    correct_df["ScanCount_abs"] = correct_df["ScanCount"].apply(lambda x: abs(x))
    temp = correct_df.groupby(["VisitNumber", "Upc"])\
                     .agg({"ScanCount": "sum"})\
                     .reset_index()

    temp = temp.rename(columns={"ScanCount": "ScanCount_sum"})
    correct_df = correct_df.merge(temp, how="left", on=["VisitNumber", "Upc"])
    temp = correct_df[(correct_df["Upc_diff"] == 0) & (correct_df["ScanCount_sum"] != 0)]

    temp["ReplacedNotExactItems"] = temp.apply(lambda row: get_replaced_notexact_items(
                                                        row["ScanCount_sum"],
                                                        row["ScanCount_abs"]), axis=1)
    temp["RefundedNotExactItems"] = temp.apply(lambda row: get_refunded_notexact_items(
                                                        row["ScanCount_sum"],
                                                        row["ScanCount_abs"]), axis=1)
    temp["ReplacedNotExactItems"] = temp.apply(lambda row: row["ReplacedNotExactItems"]
                                               if row["ReplacedNotExactItems"] != 0
                                               else row["ScanCount_abs"], axis=1)
    temp = temp[["VisitNumber", "ReplacedNotExactItems", "RefundedNotExactItems"]].groupby("VisitNumber")\
                                                                                  .agg("sum")\
                                                                                  .reset_index()
    df_new = df_new.merge(temp, how="left", on="VisitNumber")
    df_new.fillna(0, inplace=True)
    return df_new


def enum_days(df):
    """
    Enumarate the days
    """
    days = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    df["WeekdayEnum"] = df["Weekday"].apply(lambda x: days[x])
    return df


def day_index(df):
    df["DayIndex"] = df["WeekdayEnum"].rolling(2).apply(lambda x: x[0] != x[-1], raw=True).cumsum()
    df["DayIndex"] = df["DayIndex"].fillna(0)
    df["DayIndex"] = df["DayIndex"].astype(int)
    return df

def week_number(trn,tst):
    temp = trn[trn["WeekdayEnum"] == 6].groupby(["DayIndex"]).size().rolling(2).apply(lambda x: x[0] != x[-1], raw=True).cumsum()
    temp = temp.fillna(0)
    temp = pd.DataFrame({"DayIndex": temp.index, "Week": temp.values.astype(int)})
    trn = trn.merge(temp, how="left", on="DayIndex")
    tst = tst.merge(temp, how="left", on="DayIndex")
    trn["Week"] = trn['Week'].bfill()
    tst["Week"] = tst['Week'].bfill()
    return trn, tst


def no_dup_upcs(df):
    """
    Get df without the duplicate UPCs
    """
    temp1 = df[df["ScanCount"] > 0].groupby(["VisitNumber", "Upc"]).sum().reset_index()
    temp2 = df[df["ScanCount"] < 0].groupby(["VisitNumber", "Upc"]).sum().reset_index()
    df_correct = pd.concat([temp1, temp2])
    df_correct = df_correct.sort_values("VisitNumber")
    df_correct["Upc"] = df_correct["Upc"].fillna(0)
    df_correct["Upc"] = df_correct["Upc"].astype("int64")
    return df_correct


def items_bought(df, df_new):
    """
    Total items bought per Visitnumber
    """
    df = df[df["ScanCount"] >= 0].groupby("VisitNumber")["ScanCount"].sum()
    df = pd.DataFrame({"VisitNumber": df.index, "ItemsBought": df.values.astype(int)})
    df_new = df_new.merge(df,how="left", on="VisitNumber")
    df_new = df_new.fillna(0)
    return df_new


def add_triptype(df, df_new):
    temp = df.groupby(["VisitNumber", "TripType"]).first().reset_index()
    df_new = df_new.merge(temp[["TripType", "VisitNumber"]], on="VisitNumber", how="left")
    return df_new


def items_returned(df, df_new):
    """
    posa items apla epestrepsa per visit
    """
    # TODO:auto edw xreiazetai an ftaiksw swsta ta replaced refunded
    temp = df[df["ScanCount"] < 0].groupby("VisitNumber").sum().reset_index()
    temp = temp.rename({"ScanCount": "ItemsReturned"}, axis=1)
    temp["ItemsReturned"] = temp["ItemsReturned"].apply(lambda x: abs(x))

    df_new = df_new.merge(temp[["VisitNumber", "ItemsReturned"]], how="left", on="VisitNumber")
    df_new["ItemsReturned"] = df_new["ItemsReturned"].fillna(0)
    return df_new


def day_dummies(df, df_new, train=False):
    """
    get days in dummies
    """
    dummies = pd.get_dummies(df, columns=["Weekday"])
    dummies = dummies.groupby("VisitNumber").first().reset_index()
    dummies = dummies.drop(["Upc", "ScanCount", "DepartmentDescription", "FinelineNumber", "WeekdayEnum",
                                    "DayIndex", "Week"], axis=1)
    if train:
        dummies = cat_dummies_sum = dummies.drop(["TripType"], axis=1)
    df_new = df_new.merge(dummies, on="VisitNumber", how="left")
    return df_new


def entropy_test():
    arr = pd.DataFrame()
    for i in range(0,20):
        randnums = np.random.randint(1, 5, 10)
        nn = pd.Series([11, 12, 11, 10,  6, 16, 11, 11, 19,  4])

        value, counts = np.unique(randnums, return_counts=True)
        n_labels = len(randnums)
        probs = counts / n_labels
        n_classes = np.count_nonzero(probs)
        ent = entropy_calc(pd.Series(randnums), 2)
        lst = [randnums, ent, n_labels, n_classes]
        # arr = arr.append(lst)
        print(lst)
        print(entropy_calc(nn, 2))
    print()


def entropy_calc(array, base=None):
    """
    Computes entropy of label distribution.
    """
    array = array.tolist()
    n_values = len(array)

    unique_values, unique_value_counts = np.unique(array, return_counts=True)
    probs = unique_value_counts / n_values
    # n_classes = np.count_nonzero(probs)

    my_base = e if base is None else base
    return entropy(probs, base=my_base)


def fineline_entropy(df, df_new, base=None):
    temp = df.groupby("VisitNumber")["FinelineNumber"] \
             .apply(lambda x: entropy_calc(x)) \
             .reset_index()\
             .rename(columns={"FinelineNumber": "FinelineEntropy"})
    df_new = df_new.merge(temp, how="left", on="VisitNumber")
    return df_new


def fineline_chaos(df, df_new):
    temp = df.groupby("VisitNumber")["FinelineNumber"] \
             .apply(lambda x: chaos(x)) \
             .reset_index()\
             .rename(columns={"FinelineNumber": "FinelineChaos"})
    df_new = df_new.merge(temp, how="left", on="VisitNumber")
    return df_new


def chaos(array):
    """
    Computes chaos of label distribution.
    """
    array = array.tolist()
    n_values = len(array)
    unique_values, unique_value_counts = np.unique(array, return_counts=True)
    probs = unique_value_counts / n_values
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1 or n_values <= 1:
        # 1-1
        return 0
    else:
        return 1 - (entropy(probs, base=n_classes)).round(5)


def get_fav_flnumbers(df, df_new, n=3):
    """
    get favorite (most purchased) n fineline nuymbers of each visitnumber
    """
    # mipws an anti gia fill me 999 evaza to teleutiao fineline number????
    fill_na_value = -999
    temp = df.groupby(["VisitNumber", "FinelineNumber"])\
             .agg({"ScanCount": "sum"}).reset_index()

    temp = temp.groupby("VisitNumber")\
               .apply(pd.DataFrame.sort_values, ["ScanCount"], ascending=False)\
               .reset_index(drop=True)
    temp = temp.groupby("VisitNumber").head(n)
    temp1 = temp.groupby("VisitNumber") \
                .nth(0)\
                .reset_index()\
                .rename(columns={"FinelineNumber": "FavFlNo1",
                                 "ScanCount": "FavFlNo1Items"})
    for i in range(1, n):
        print()
        temp1 = temp1.merge(temp.groupby("VisitNumber")
                                .nth(i)
                                .reset_index()
                                .rename(columns={"FinelineNumber": "FavFlNo"+str(i+1),
                                                 "ScanCount": "FavFlNo"+str(i+1)+"Items"}),
                            how="left",
                            on="VisitNumber")

    df_new = df_new.merge(temp1, how="left", on="VisitNumber")
    for i in range(0, n):
        df_new["FavFlNo"+str(i+1)] = df_new["FavFlNo"+str(i+1)].fillna(fill_na_value)
        df_new["FavFlNo" + str(i + 1)] = df_new["FavFlNo" + str(i + 1)].apply(lambda x: abs(x))
        df_new["FavFlNo"+str(i+1)+"Items"] = df_new["FavFlNo"+str(i+1)+"Items"].fillna(0)
        df_new["FavFlNo" + str(i + 1) + "Items"] = df_new["FavFlNo" + str(i + 1) + "Items"].apply(lambda x: abs(x))
    return df_new


def fineline_dummies(df, df_new, add_perc=False):

    fineline_dum = pd.get_dummies(df[["VisitNumber", "FinelineNumber"]], columns=["FinelineNumber"],prefix="FLNumber", prefix_sep="_")
    fineline_dum = fineline_dum.groupby("VisitNumber").sum().reset_index()
    fineline_dum.to_csv(PATH + "/temp/" + "fineline_dum.csv", index=False)
    cols = fineline_dum.columns[1:]
    df_new = df_new.merge(fineline_dum, how="left", on="VisitNumber")
    if add_perc:
        for col in cols:
            df_new[col + "_perc"] = (df_new[col]/df_new["TotalItems"]).round(3)

    df_new.to_csv(PATH + "/temp/" + "new_df_temp.csv", index=False)
    return df_new
    print()


def days_sin_cos(df_new):
    """
    convert days into sin and cos so it becomes cyclical.
    """
    df_new['sin_day'] = np.sin(2 * np.pi * df_new["WeekdayEnum"] / 7)
    df_new['cos_day'] = np.cos(2 * np.pi * df_new["WeekdayEnum"] / 7)
    df_new = df_new.drop(['WeekdayEnum'], axis=1)
    return df_new


def category_dummies(df, df_new, train=False, add_perc=False):
    """
    Get categories in dummies
    """
    cat_dummies_sum = pd.get_dummies(df, columns=["DepartmentDescription"], prefix="", prefix_sep="")
    cat_dummies_sum = cat_dummies_sum.groupby("VisitNumber").sum().reset_index()
    cat_dummies_sum = cat_dummies_sum.drop(["Week", "WeekdayEnum", "Upc", "ScanCount", "FinelineNumber"], axis=1)
    if train:
        cat_dummies_sum = cat_dummies_sum.drop(["TripType"], axis=1)
        cols = cat_dummies_sum.columns[2:]
    else:
        cols = cat_dummies_sum.columns[1:]
    df_new = df_new.merge(cat_dummies_sum, how="left", on="VisitNumber")
    if add_perc:
        for col in cols:
            df_new[col + "_perc"] = (df_new[col] /df_new["ItemsBought"]).round(3)

    return df_new


def nominal_encode(column, fit_column=None, get_mappings=False, le_mapping=None, ):
    le = preprocessing.LabelEncoder()
    if fit_column:
        le.fit(fit_column)
    else:
        le.fit(column)

    if le_mapping == None:
        le_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        le_mapping_inv = dict(zip(le.transform(le.classes_), le.classes_))
    if get_mappings:
        return column.map(le_mapping), le_mapping, le_mapping_inv
    return column.map(le_mapping)


departmentGenCat2 = {'FINANCIAL SERVICES': "Services",
                     'SHOES': "WEARABLES",
                     'PERSONAL CARE': "RECURRING",
                     'PAINT AND ACCESSORIES': "PLANNED",
                     'DSD GROCERY' 'MEAT - FRESH & FROZEN': "RECURRING",
                     'DAIRY': "RECURRING",
                     'PETS AND SUPPLIES': "RECURRING",
                     'HOUSEHOLD CHEMICALS/SUPP': "RECURRING",
                     'IMPULSE MERCHANDISE': "IMPULSE",
                     'PRODUCE': "RECURRING",
                     'CANDY, TOBACCO, COOKIES': "RECURRING",
                     'GROCERY DRY GOODS': "RECURRING",
                     'BOYS WEAR': "CLOTHES",
                     'FABRICS AND CRAFTS': "PLANNED",
                     'JEWELRY AND SUNGLASSES': "PLANNED",
                     'MENS WEAR': "PLANNED",
                     'ACCESSORIES': "PLANNED",
                     'HOME MANAGEMENT': "PLANNED",
                     'FROZEN FOODS': "RECURRING",
                     'SERVICE DELI': "RECURRING",
                     'INFANT CONSUMABLE HARDLINES': "RECURRING",
                     'PRE PACKED DELI': "RECURRING",
                     'COOK AND DINE': "RECURRING",
                     'PHARMACY OTC': "RECURRING",
                     'LADIESWEAR': "PLANNED",
                     'COMM BREAD': "RECURRING",
                     'BAKERY': "RECURRING",
                     'HOUSEHOLD PAPER GOODS': "RECURRING",
                     'CELEBRATION': "PLANNED",
                     'HARDWARE': "PLANNED",
                     'BEAUTY': "RECURRING",
                     'AUTOMOTIVE': "PLANNED",
                     'BOOKS AND MAGAZINES': "IMPULSE",
                     'SEAFOOD': "RECURRING",
                     'OFFICE SUPPLIES': "RECURRING",
                     'LAWN AND GARDEN': "RECURRING",
                     'SHEER HOSIERY': "RECURRING",
                     'WIRELESS': "PLANNED",
                     'BEDDING': "PLANNED",
                     'BATH AND SHOWER': "RECURRING",
                     'HORTICULTURE AND ACCESS': "PLANNED",
                     'HOME DECOR': "PLANNED",
                     'TOYS': "IMPULSE",
                     'INFANT APPAREL': "WEARBLES",
                     'LADIES SOCKS': "WEARABLES",
                     'PLUS AND MATERNITY': "PLANNED",
                     'ELECTRONICS': "PLANNED",
                     'GIRLS WEAR, 4-6X  AND 7-14': "WEARABLES",
                     'BRAS & SHAPEWEAR': "WEARABLES",
                     'LIQUOR,WINE,BEER': "RECURRING",
                     'SLEEPWEAR/FOUNDATIONS': "WEARABLES",
                     'CAMERAS AND SUPPLIES': "PLANNED",
                     'SPORTING GOODS': "PLANNED",
                     'PLAYERS AND ELECTRONICS': "PLANNED",
                     'PHARMACY RX': "RECURRING",
                     'MENSWEAR': "PLANNED",
                     'OPTICAL - FRAMES': "PLANNED",
                     'SWIMWEAR/OUTERWEAR': "WEARABLES",
                     'OTHER DEPARTMENTS': "OTHER",
                     'MEDIA AND GAMING': "PLANNED",
                     'FURNITURE': "PLANNED",
                     'OPTICAL - LENSES': "RECURRING",
                     'SEASONAL': "OTHER",
                     'LARGE HOUSEHOLD GOODS': "PLANNED",
                     '1-HR PHOTO': "PLANNED",
                     'CONCEPT STORES': "OTHER",
                     'HEALTH AND BEAUTY AIDS': "RECURRING",
                     }

departmentGenCat1 = {'FINANCIAL SERVICES': "OTHER",
                     'SHOES': "WEARABLES",
                     'PERSONAL CARE': "PERSONAL CARE",
                     'PAINT AND ACCESSORIES': "TOOLS",
                     'DSD GROCERY' 'MEAT - FRESH & FROZEN': "FOOD",
                     'DAIRY': "FOOD",
                     'PETS AND SUPPLIES': "FOOD",
                     'HOUSEHOLD CHEMICALS/SUPP': "HOMESTUFF",
                     'IMPULSE MERCHANDISE': "IMPULSE",
                     'PRODUCE': "FOOD",
                     'CANDY, TOBACCO, COOKIES': "FOOD",
                     'GROCERY DRY GOODS': "FOOD",
                     'BOYS WEAR': "CLOTHES",
                     'FABRICS AND CRAFTS': "WEARABLES",
                     'JEWELRY AND SUNGLASSES': "WEARABLES",
                     'MENS WEAR': "WEARABLES",
                     'ACCESSORIES': "WEARABLES",
                     'HOME MANAGEMENT': "HOMESTUFF",
                     'FROZEN FOODS': "FOOD",
                     'SERVICE DELI': "FOOD",
                     'INFANT CONSUMABLE HARDLINES': "FUN",
                     'PRE PACKED DELI': "FOOD",
                     'COOK AND DINE': "FOOD",
                     'PHARMACY OTC': "PHARMACY",
                     'LADIESWEAR': "WEARABLES",
                     'COMM BREAD': "FOOD",
                     'BAKERY': "FOOD",
                     'HOUSEHOLD PAPER GOODS': "HOMESTUFF",
                     'CELEBRATION': "FUN",
                     'HARDWARE': "HOMESTUFF",
                     'BEAUTY': "PERSONAL CARE",
                     'AUTOMOTIVE': "HOMESTUFF",
                     'BOOKS AND MAGAZINES': "FUN",
                     'SEAFOOD': "FOOD",
                     'OFFICE SUPPLIES': "HOMESTUFF",
                     'LAWN AND GARDEN': "HOMESTUFF",
                     'SHEER HOSIERY': "WEARABLES",
                     'WIRELESS': "ELECTRONICS",
                     'BEDDING': "HOMESTUFF",
                     'BATH AND SHOWER': "HOMESTUFF",
                     'HORTICULTURE AND ACCESS': "HOMESTUFF",
                     'HOME DECOR': "HOMESTUFF",
                     'TOYS': "FUN",
                     'INFANT APPAREL': "WEARBLES",
                     'LADIES SOCKS': "WEARABLES",
                     'PLUS AND MATERNITY': "CLOTHES",
                     'ELECTRONICS': "ELECTRONICS",
                     'GIRLS WEAR, 4-6X  AND 7-14': "WEARABLES",
                     'BRAS & SHAPEWEAR': "WEARABLES",
                     'LIQUOR,WINE,BEER': "FOOD",
                     'SLEEPWEAR/FOUNDATIONS': "WEARABLES",
                     'CAMERAS AND SUPPLIES': "ELECTRONICS",
                     'SPORTING GOODS': "FUN",
                     'PLAYERS AND ELECTRONICS': "ELECTRONICS",
                     'PHARMACY RX': "PHARMACY",
                     'MENSWEAR': "WEARABLES",
                     'OPTICAL - FRAMES': "WEARABLES",
                     'SWIMWEAR/OUTERWEAR': "WEARABLES",
                     'OTHER DEPARTMENTS': "",
                     'MEDIA AND GAMING': "ELECTRONICS",
                     'FURNITURE': "HOMESTUFF",
                     'OPTICAL - LENSES': "WEARABLES",
                     'SEASONAL': "OTHER",
                     'LARGE HOUSEHOLD GOODS': "HOMESTUFF",
                     '1-HR PHOTO': "OTHER",
                     'CONCEPT STORES': "OTHER",
                     'HEALTH AND BEAUTY AIDS': "PERSONAL CARE",
                     }
# train.to_csv(PATH + "/temp/" + "train_temp.csv", index=False)
# test.to_csv(PATH + "/temp/" + "test_temp.csv", index=False)
# merged.to_csv(PATH + "/temp/" + "merged_temp.csv", index=False)


