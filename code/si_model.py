import matplotlib.pyplot as plt
import pandas as pd
from source import compile_data

def extract_SI_data(data, params):
    num_days = params['R_time']
    columns_to_extract = ['date','week','year','population','total_cases','new_cases','total_deaths','new_deaths','people_fully_vaccinated',
                          'reproduction_rate','avg_USA']
    SI_data = data[columns_to_extract]

    # first_vaccination_idx = SI_data[SI_data['people_fully_vaccinated'].notna()].index[0]
    # date_of_first_vaccination = SI_data.loc[first_vaccination_idx, 'date'].date()
    # print(f'First day with vaccination data: {date_of_first_vaccination}')
    
    SI_data = SI_data.fillna(0)
    SI_data.rename(columns={'total_cases':'I','new_cases':'I_daily','total_deaths': 'D','new_deaths':'D_daily','population':'N',
                               'reproduction_rate':'R_0','people_fully_vaccinated': 'V','avg_USA':'mobility_index'}, inplace=True)
    SI_data.loc[1:,'N'] = SI_data['N'].shift(1) - SI_data['D']

    SI_data['I_pop'] = SI_data['I'] - SI_data['D']
    # People alive and infected R_time days earlier will become susceptible again
    SI_data.loc[num_days:,'I_pop'] = SI_data['I_pop'] - SI_data['I_pop'].shift(params['R_time'])
    SI_data['S_pop'] = (SI_data['N'] - SI_data['I_pop'])

    SI_data['S'] = SI_data['S_pop'] / SI_data['N']
    SI_data['I'] = SI_data['I_pop'] / SI_data['N']
    SI_data['V'] = SI_data['V'] / SI_data['N']
    SI_data.drop(['S_pop', 'I_pop'], axis=1, inplace=True)

    columns_to_extract = ['date','week','year','N','S','I','V','I_daily','D_daily','R_0','mobility_index']
    SI_model_data = SI_data[columns_to_extract]
    return SI_model_data

def model_params():
    params = {'R_time':14}
    return params

def SI_plot(col, SI_data, save_path):
    unique_years = SI_data['year'].unique()
    for year in unique_years:
        data_year = SI_data[SI_data['year'] == year]
        weekly_avg_data = data_year.groupby(['year', 'week']).agg({col: 'mean'}).reset_index()
        weekly_avg_data['year_week'] = weekly_avg_data['year'].astype(str) + ' - W' + weekly_avg_data['week'].astype(str)
        plt.figure(figsize=(12, 6))
        plt.xlabel('Year-Week')
        plt.ylabel('Population')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        if col == 'S':
            plt.plot(weekly_avg_data['year_week'], weekly_avg_data[col], label='Susceptible (S)', color='g', marker='.', linestyle='--')
            plt.title(f'Susceptible Population Proportion (S) over Year-Week for {year}')
            path = save_path + f'S_plot_{year}.png'
        elif col == 'I':
            plt.plot(weekly_avg_data['year_week'], weekly_avg_data[col], label='Susceptible (I)', color='r', marker='.', linestyle='--')
            plt.title(f'Infected Population Proportion (I) over Year-Week for {year}')
            path = save_path + f'I_plot_{year}.png'
        plt.savefig(path)
        plt.show()

if __name__ == "__main__":
    data = compile_data()
    params = model_params()
    SI_data = extract_SI_data(data, params)
    save_path = '../plots/SI_model/'
    for col in ['S', 'I']:
        SI_plot(col, SI_data, save_path)