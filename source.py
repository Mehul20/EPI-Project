import pandas as pd

def read_data(filePath):
    data = pd.read_csv(filePath)
    return data

def get_USA_data(data):
    return data[data['iso_code'] == 'USA']

def remove_columns(data):
    columns_to_drop = ['iso_code','continent','location','new_cases_smoothed','new_deaths_smoothed','total_cases_per_million',
                       'new_cases_per_million','new_cases_smoothed_per_million','total_deaths_per_million','new_deaths_per_million',
                       'new_deaths_smoothed_per_million','icu_patients_per_million','hosp_patients_per_million','weekly_icu_admissions_per_million',
                       'weekly_hosp_admissions_per_million','total_tests_per_thousand','new_tests_per_thousand','new_tests_smoothed',
                       'new_tests_smoothed_per_thousand','new_vaccinations_smoothed','total_vaccinations_per_hundred','people_vaccinated_per_hundred',
                       'people_fully_vaccinated_per_hundred','total_boosters_per_hundred','new_vaccinations_smoothed_per_million',
                       'new_people_vaccinated_smoothed','new_people_vaccinated_smoothed_per_hundred','aged_65_older','aged_70_older',
                       'hospital_beds_per_thousand', 'excess_mortality_cumulative_absolute','excess_mortality_cumulative_per_million']

    # Columns not removed from dataframe!
    # ---------------------------------------------------------------------------------------------------------------------------------------------------
    # date,total_cases,new_cases,total_deaths,new_deaths,reproduction_rate,icu_patients,hosp_patients,weekly_icu_admissions,weekly_hosp_admissions,
    # total_tests,new_tests,positive_rate,tests_per_case,tests_units,total_vaccinations,people_vaccinated,people_fully_vaccinated,total_boosters,
    # new_vaccinations,stringency_index,population_density,median_age,gdp_per_capita,extreme_poverty,cardiovasc_death_rate,diabetes_prevalence,
    # female_smokers,male_smokers,handwashing_facilities,life_expectancy,human_development_index,population,excess_mortality_cumulative,excess_mortality,
    # ---------------------------------------------------------------------------------------------------------------------------------------------------
    data = data.drop(columns_to_drop, axis=1)
    return data

if __name__ == "__main__":
    filePath = "cowid-covid-data.csv"
    data = read_data(filePath)
    US_data = get_USA_data(data)
    filtered_data = remove_columns(US_data)
    print(filtered_data)