import pandas as pd
import numpy as np


def get_imputation_values(df_filtered: pd.DataFrame,):
    """creates imputation values which are not rule based and returns them

    Args:
        df_filtered (pd.DataFrame): filtered Dataframe, should not include any validation/test data, is used to create the non rule based imputation values

    Returns:
        dict_impute_values:dict: not rule based imputation values
    """
    # region Testosteron
    # impute values
    # male median separated by gonado disfunction
    # female just the median, should be low <= 3
    testo_median_male_gonado_yes = df_filtered[(df_filtered['Pre_OP_hormone_gonado'] == True) & (
        df_filtered['Patient_gender'] == 'male')]['TEST'].median()
    testo_median_male_gonado_no = df_filtered[(df_filtered['Pre_OP_hormone_gonado'] == False) & (
        df_filtered['Patient_gender'] == 'male')]['TEST'].median()
    testo_median_female = df_filtered[df_filtered['Patient_gender']
                                      == 'female']['TEST'].median()
    # endregion

    # region LH
    # LH
    # male median separated by gonado disfunction
    # female by gonado disfunction and separation age > 51, indictating menopause
    lh_median_male_gonado_yes = df_filtered[(df_filtered['Pre_OP_hormone_gonado'] == True) & (
        df_filtered['Patient_gender'] == 'male')]['LH'].median()
    lh_median_male_gonado_no = df_filtered[(df_filtered['Pre_OP_hormone_gonado'] == False) & (
        df_filtered['Patient_gender'] == 'male')]['LH'].median()

    lh_median_female_gonado_yes = df_filtered[(df_filtered['Patient_age'] <= 51) & (
        df_filtered['Pre_OP_hormone_gonado'] == True) & (df_filtered['Patient_gender'] == 'female')]['LH'].median()
    lh_median_female_gonado_yes_menopause = df_filtered[(df_filtered['Patient_age'] > 51) & (
        df_filtered['Pre_OP_hormone_gonado'] == True) & (df_filtered['Patient_gender'] == 'female')]['LH'].median()

    lh_median_female_gonado_no = df_filtered[(df_filtered['Patient_age'] <= 51) & (
        df_filtered['Pre_OP_hormone_gonado'] == False) & (df_filtered['Patient_gender'] == 'female')]['LH'].median()
    lh_median_female_gonado_no_menopause = df_filtered[(df_filtered['Patient_age'] > 51) & (
        df_filtered['Pre_OP_hormone_gonado'] == False) & (df_filtered['Patient_gender'] == 'female')]['LH'].median()
    # endregion

    # region FSH
    # FSH
    # male median separated by gonado disfunction
    # female by gonado disfunction and separation age > 51, indictating menopause
    fsh_median_male_gonado_yes = df_filtered[(df_filtered['Pre_OP_hormone_gonado'] == True) & (
        df_filtered['Patient_gender'] == 'male')]['FSH'].median()
    fsh_median_male_gonado_no = df_filtered[(df_filtered['Pre_OP_hormone_gonado'] == False) & (
        df_filtered['Patient_gender'] == 'male')]['FSH'].median()

    fsh_median_female_gonado_yes = df_filtered[(df_filtered['Patient_age'] <= 51) & (
        df_filtered['Pre_OP_hormone_gonado'] == True) & (df_filtered['Patient_gender'] == 'female')]['FSH'].median()
    fsh_median_female_gonado_yes_menopause = df_filtered[(df_filtered['Patient_age'] > 51) & (
        df_filtered['Pre_OP_hormone_gonado'] == True) & (df_filtered['Patient_gender'] == 'female')]['FSH'].median()

    fsh_median_female_gonado_no = df_filtered[(df_filtered['Patient_age'] <= 51) & (
        df_filtered['Pre_OP_hormone_gonado'] == False) & (df_filtered['Patient_gender'] == 'female')]['FSH'].median()
    fsh_median_female_gonado_no_menopause = df_filtered[(df_filtered['Patient_age'] > 51) & (
        df_filtered['Pre_OP_hormone_gonado'] == False) & (df_filtered['Patient_gender'] == 'female')]['FSH'].median()
    # endregion

    dict_impute_values = {'testo_median_male_gonado_yes': testo_median_male_gonado_yes,
                          'testo_median_male_gonado_no': testo_median_male_gonado_no,
                          'testo_median_female': testo_median_female,

                          'lh_median_male_gonado_yes': lh_median_male_gonado_yes,
                          'lh_median_male_gonado_no': lh_median_male_gonado_no,
                          'lh_median_female_gonado_yes': lh_median_female_gonado_yes,
                          'lh_median_female_gonado_yes_menopause': lh_median_female_gonado_yes_menopause,
                          'lh_median_female_gonado_no': lh_median_female_gonado_no,
                          'lh_median_female_gonado_no_menopause': lh_median_female_gonado_no_menopause,

                          'fsh_median_male_gonado_yes': fsh_median_male_gonado_yes,
                          'fsh_median_male_gonado_no': fsh_median_male_gonado_no,
                          'fsh_median_female_gonado_yes': fsh_median_female_gonado_yes,
                          'fsh_median_female_gonado_yes_menopause': fsh_median_female_gonado_yes_menopause,
                          'fsh_median_female_gonado_no': fsh_median_female_gonado_no,
                          'fsh_median_female_gonado_no_menopause': fsh_median_female_gonado_no_menopause,

                          }
    return dict_impute_values


def impute_dataframe(df_to_impute: pd.DataFrame, impute_values: dict):
    """imputes a dataframe with the impute values

    Args:
        df_to_impute (pd.DataFrame): dataframe which should be imputed with values
        impute_values (dict): impute values, dictionary with values to impute, should be created by get_imputation_values function

    Returns:
        df_to_impute: imputed dataframe
    """

    age_gender_ranges_test = [
        {'age_range': (df_to_impute['Patient_age'].min(), df_to_impute['Patient_age'].max()),
         'gender': 'male',
         'imputation_value': impute_values['testo_median_male_gonado_yes'],
         'Pre_OP_hormone_gonado': True},
        {'age_range': (df_to_impute['Patient_age'].min(), df_to_impute['Patient_age'].max()),
         'gender': 'male',
         'imputation_value': impute_values['testo_median_male_gonado_no'],
         'Pre_OP_hormone_gonado': False},
        {'age_range': (df_to_impute['Patient_age'].min(), df_to_impute['Patient_age'].max()),
         'gender': 'female',
         'imputation_value': impute_values['testo_median_female'],
         'Pre_OP_hormone_gonado': False},
        {'age_range': (df_to_impute['Patient_age'].min(), df_to_impute['Patient_age'].max()),
         'gender': 'female',
         'imputation_value': impute_values['testo_median_female'],
         'Pre_OP_hormone_gonado': True}
    ]

    age_gender_ranges_lh = [
        {'age_range': (df_to_impute['Patient_age'].min(), df_to_impute['Patient_age'].max()),
         'gender': 'male',
         'imputation_value': impute_values['lh_median_male_gonado_yes'],
         'Pre_OP_hormone_gonado': True},
        {'age_range': (df_to_impute['Patient_age'].min(), df_to_impute['Patient_age'].max()),
         'gender': 'male',
         'imputation_value': impute_values['lh_median_male_gonado_no'],
         'Pre_OP_hormone_gonado': False},

        {'age_range': (df_to_impute['Patient_age'].min(), 51),
         'gender': 'female',
         'imputation_value': impute_values['lh_median_female_gonado_yes'],
         'Pre_OP_hormone_gonado': True},
        {'age_range': (52, df_to_impute['Patient_age'].max()),
         'gender': 'female',
         'imputation_value': impute_values['lh_median_female_gonado_yes_menopause'],
         'Pre_OP_hormone_gonado': True},

        {'age_range': (df_to_impute['Patient_age'].min(), 51),
         'gender': 'female',
         'imputation_value': impute_values['lh_median_female_gonado_no'],
         'Pre_OP_hormone_gonado': False},
        {'age_range': (52, df_to_impute['Patient_age'].max()),
         'gender': 'female',
         'imputation_value': impute_values['lh_median_female_gonado_no_menopause'],
         'Pre_OP_hormone_gonado': False}
    ]
    age_gender_ranges_fsh = [
        {'age_range': (df_to_impute['Patient_age'].min(), df_to_impute['Patient_age'].max()),
         'gender': 'male',
         'imputation_value': impute_values['fsh_median_male_gonado_yes'],
         'Pre_OP_hormone_gonado': True},
        {'age_range': (df_to_impute['Patient_age'].min(), df_to_impute['Patient_age'].max()),
         'gender': 'male',
         'imputation_value': impute_values['fsh_median_male_gonado_no'],
         'Pre_OP_hormone_gonado': False},

        {'age_range': (df_to_impute['Patient_age'].min(), 51),
         'gender': 'female',
         'imputation_value': impute_values['fsh_median_female_gonado_yes'],
         'Pre_OP_hormone_gonado': True},
        {'age_range': (52, df_to_impute['Patient_age'].max()),
         'gender': 'female',
         'imputation_value': impute_values['fsh_median_female_gonado_yes_menopause'],
         'Pre_OP_hormone_gonado': True},

        {'age_range': (df_to_impute['Patient_age'].min(), 51),
         'gender': 'female',
         'imputation_value': impute_values['fsh_median_female_gonado_no'],
         'Pre_OP_hormone_gonado': False},
        {'age_range': (52, df_to_impute['Patient_age'].max()),
         'gender': 'female',
         'imputation_value': impute_values['fsh_median_female_gonado_no_menopause'],
         'Pre_OP_hormone_gonado': False}
    ]

    # we impute the median of our population
    df_to_impute['TEST'] = df_to_impute.apply(_impute_based_on_age_gender_disfunction, args=(
        ['TEST', age_gender_ranges_test, 'Pre_OP_hormone_gonado']), axis=1)
    # we impute the mean of our trainingset population, we differentiate by gender
    df_to_impute['LH'] = df_to_impute.apply(_impute_based_on_age_gender_disfunction, args=(
        ['LH', age_gender_ranges_lh, 'Pre_OP_hormone_gonado']), axis=1)
    # we impute the mean of our trainingset population, we differentiate by gender
    df_to_impute['FSH'] = df_to_impute.apply(_impute_based_on_age_gender_disfunction, args=(
        ['FSH', age_gender_ranges_fsh, 'Pre_OP_hormone_gonado']), axis=1)

    # imputa all other rule based values
    df_to_impute = _impute_rule_based(df_to_impute)

    return df_to_impute


def _impute_rule_based(df_to_impute: pd.DataFrame):

    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5794073/ - Table 3 - Mean
    age_ranges_ft4 = [(df_to_impute['Patient_age'].min(), 19, 1.30*12.871), (20, 29, 1.31*12.871), (30, 39, 1.26*12.871),
                      (40, 49, 1.22*12.871), (50, 59, 1.20*12.871), (60, 69, 1.2*12.871), (70, df_to_impute['Patient_age'].max(), 1.2*12.871)]
    df_to_impute['FT4'] = df_to_impute.apply(
        _impute_based_on_age, args=(['FT4', age_ranges_ft4]), axis=1)

    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9206165/ - Table 1 - Median
    age_ranges_igf1 = [(df_to_impute['Patient_age'].min(), 25, 265.00*0.131), (26, 30, 222*0.131), (31, 35, 183*0.131),
                       (36, 40, 171*0.131), (41, 45,
                                             148*0.131), (46, 50, 129*0.131),
                       (51, 55, 129.5*0.131), (56, 60, 130*0.131), (61, 65, 129.5*0.131), (66, 70, 128*0.131), (71, 75, 123*0.131), (76, df_to_impute['Patient_age'].max(), 109*0.131)]
    df_to_impute['IGF1'] = df_to_impute.apply(
        _impute_based_on_age, args=(['IGF1', age_ranges_igf1]), axis=1)

    # regular value for cortisol is at least 550 nmol/l
    df_to_impute['COR'] = df_to_impute['COR'].fillna(550)

    # https://labosud.fr/wp-content/uploads/sites/7/2023/08/Verification_of_Roche_reference_ranges_for_serum_prolactin_in_children_adolescents_adults_and_the_elderly-2023-07-11-05-21.pdf - Table 2 - Median by age and gender
    age_gender_ranges_prol = [
        {'age_range': (df_to_impute['Patient_age'].min(
        ), 9), 'gender': 'male', 'imputation_value': 194.8},
        {'age_range': (10, 12), 'gender': 'male', 'imputation_value': 273.9},
        {'age_range': (13, 16), 'gender': 'male', 'imputation_value': 352.6},
        {'age_range': (17, 19), 'gender': 'male', 'imputation_value': 397.},
        {'age_range': (20, 30), 'gender': 'male', 'imputation_value': 356.5},
        {'age_range': (31, 40), 'gender': 'male', 'imputation_value': 325.4},
        {'age_range': (41, 50), 'gender': 'male', 'imputation_value': 292.9},
        {'age_range': (51, 60), 'gender': 'male', 'imputation_value': 272.4},
        {'age_range': (61, 70), 'gender': 'male', 'imputation_value': 227.9},
        {'age_range': (71, df_to_impute['Patient_age'].max(
        )), 'gender': 'male', 'imputation_value': 250.3},

        {'age_range': (df_to_impute['Patient_age'].min(
        ), 10), 'gender': 'female', 'imputation_value': 211.2},
        {'age_range': (11, 13), 'gender': 'female', 'imputation_value': 211.2},
        {'age_range': (14, 16), 'gender': 'female', 'imputation_value': 262.0},
        {'age_range': (17, 19), 'gender': 'female', 'imputation_value': 283.8},
        {'age_range': (20, 30), 'gender': 'female', 'imputation_value': 291.7},
        {'age_range': (31, 40), 'gender': 'female', 'imputation_value': 260.5},
        {'age_range': (41, 50), 'gender': 'female', 'imputation_value': 252.5},
        {'age_range': (51, 60), 'gender': 'female', 'imputation_value': 241.},
        {'age_range': (61, 70), 'gender': 'female', 'imputation_value': 232.2},
        {'age_range': (71, df_to_impute['Patient_age'].max(
        )), 'gender': 'female', 'imputation_value': 252.4},
    ]

    df_to_impute['PROL'] = df_to_impute.apply(
        _impute_based_on_age_gender, args=(['PROL', age_gender_ranges_prol]), axis=1)

    return df_to_impute


def _impute_based_on_age(row, column, age_ranges):
    age = row['Patient_age']
    col_value = row[column]
    if pd.isna(col_value):
        for start, end, imputation_value in age_ranges:
            if start <= age <= end:
                return imputation_value
        return np.nan
    else:
        return col_value


def _impute_based_on_age_gender(row, column, age_gender_ranges):
    age = row['Patient_age']
    gender = row['Patient_gender']
    col_value = row[column]
    if pd.isna(col_value):
        for age_range in age_gender_ranges:
            if (age_range['age_range'][0] <= age <= age_range['age_range'][1]) and age_range['gender'] == gender:
                return age_range['imputation_value']
    return col_value


def _impute_based_on_age_gender_disfunction(row, column, age_gender_ranges, boolean_condition_column):
    age = row['Patient_age']
    gender = row['Patient_gender']
    col_value = row[column]

    if pd.isna(col_value):
        for age_range in age_gender_ranges:
            condition_match = (
                (age_range['age_range'][0] <= age <= age_range['age_range'][1]) and
                age_range['gender'] == gender and
                row[boolean_condition_column] == age_range[boolean_condition_column]
            )

            if condition_match:
                return age_range['imputation_value']

    return col_value
