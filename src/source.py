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

    # Columns not removed from dataframe
    # ---------------------------------------------------------------------------------------------------------------------------------------------------
    # date,total_cases,new_cases,total_deaths,new_deaths,reproduction_rate,icu_patients,hosp_patients,weekly_icu_admissions,weekly_hosp_admissions,
    # total_tests,new_tests,positive_rate,tests_per_case,tests_units,total_vaccinations,people_vaccinated,people_fully_vaccinated,total_boosters,
    # new_vaccinations,stringency_index,population_density,median_age,gdp_per_capita,extreme_poverty,cardiovasc_death_rate,diabetes_prevalence,
    # female_smokers,male_smokers,handwashing_facilities,life_expectancy,human_development_index,population,excess_mortality_cumulative,excess_mortality
    # ---------------------------------------------------------------------------------------------------------------------------------------------------
    data = data.drop(columns_to_drop, axis=1)
    return data

def add_week_year(data):
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    data['week'] = data['date'].dt.isocalendar().week
    return data

def remove_years_2023_2024(data):
    filtered_data = data[~data['year'].isin([2023, 2024])]
    return filtered_data

def clean_case_data():
    filePath = "../data/cowid-covid-data.csv"
    case_data = read_data(filePath)
    US_data = get_USA_data(case_data)
    filtered_case_data_1 = remove_columns(US_data)
    filtered_case_data_2 = add_week_year(filtered_case_data_1)
    filtered_case_data_3 = remove_years_2023_2024(filtered_case_data_2)
    return filtered_case_data_3

def clean_twitter_mobility_data():
    filePath = "../data/mobility.csv"
    mobility_data = read_data(filePath)
    mobility_data = mobility_data.rename(columns={'Dates': 'date'})
    filtered_case_data_1 = add_week_year(mobility_data)
    return filtered_case_data_1

def merge_and_clean_data(case_data, mobility_data):
    # master_data_initial will be used in the future when we are working with state/city data.
    master_data_initial = pd.merge(case_data, mobility_data, on = ["year", "week"], how = "left")
    
    locations_to_drop = [
        "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN", "KS", "KY", 
        "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", 
        "OK", "OR", "PA", "PR", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VI", "VT", "WA", "WI", "WV", "WY", 
        "Northeast", "Midwest", "Central", "South", "West", "Caribbean", 
        "Albuquerque", "Anaheim", "Anchorage", "Arlington", "Atlanta", "Aurora", "Austin", "Bakersfield", 
        "Baltimore", "Baton Rouge", "Boise", "Boston", "Buffalo", "Chandler", "Charlotte", "Chesapeake", 
        "Chicago", "Chula Vista", "Cincinnati", "Cleveland", "Colorado Springs", "Columbus", "Corpus Christi", 
        "Dallas", "Denver", "Detroit", "Durham", "El Paso", "Fort Wayne", "Fort Worth", "Fremont", "Fresno", 
        "Garland", "Gilbert", "Glendale", "Greensboro", "Henderson", "Hialeah", "Honolulu", "Houston", 
        "Indianapolis", "Irvine", "Irving", "Jacksonville", "Jersey City", "Kansas City", "Laredo", 
        "Las Vegas", "Lexington", "Lincoln", "Long Beach", "Los Angeles", "Louisville", "Lubbock", "Madison", 
        "Memphis", "Mesa", "Miami", "Milwaukee", "Minneapolis", "Nashville", "New York City", "New Orleans", 
        "Newark", "Norfolk", "North Las Vegas", "Oakland", "Oklahoma City", "Omaha", "Orlando", "Philadelphia", 
        "Phoenix", "Pittsburgh", "Plano", "Portland", "Raleigh", "Reno", "Richmond", "Riverside", "Sacramento", 
        "San Antonio", "San Diego", "San Francisco", "San Jose", "Santa Ana", "Scottsdale", "Seattle", "Spokane", 
        "St Louis", "St Paul", "St Petersburg", "Stockton", "Tampa", "Toledo", "Tucson", "Tulsa", "Virginia Beach", 
        "Washington", "Wichita", "Winston-Salem", "date_y"
    ]

    master_data = master_data_initial.drop(locations_to_drop, axis=1)
    master_data = master_data.rename(columns={'date_x': 'date'})
    return master_data

def compile_data():
    case_data = clean_case_data()
    mobility_data = clean_twitter_mobility_data()
    master_data = merge_and_clean_data(case_data, mobility_data)
    master_data.to_csv('../data/master_data.csv', index=False)
    return master_data

if __name__ == "__main__":
    compile_data()