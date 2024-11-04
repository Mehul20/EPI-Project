from source import compile_data

def extract_SI_data(data, params):
    num_days = params['R_time']
    print(num_days)
    columns_to_extract = ['date','population','total_cases','new_cases','total_deaths','new_deaths','people_fully_vaccinated',
                          'reproduction_rate','avg_USA']
    SI_data = data[columns_to_extract]

    first_vaccination_idx = SI_data[SI_data['people_fully_vaccinated'].notna()].index[0]
    date_of_first_vaccination = SI_data.loc[first_vaccination_idx, 'date'].date()
    # print(f'First day with vaccination data: {date_of_first_vaccination}')
    SI_data = SI_data.fillna(0)
    SI_data.rename(columns={'total_cases':'I','new_cases':'I_daily','total_deaths': 'D','new_deaths':'D_daily','population':'N',
                               'reproduction_rate':'R_0','people_fully_vaccinated': 'V','avg_USA':'mobility_index'}, inplace=True)
    SI_data.loc[1:,'N'] = SI_data['N'].shift(1) - SI_data['D']

    I_pop_vals = SI_data['I']
    SI_data.insert(6, 'I_pop', I_pop_vals)
    # People alive and infected R_time days earlier will become susceptible again
    # SI_data.loc[num_days:,'I_pop'] = SI_data['I_pop'] - SI_data['I'].shift(params['R_time']) - SI_data['I'].shift(params['R_time'])\
    #     + SI_data['D'] - SI_data['D'].shift(params['R_time'])
    S_vals = SI_data['N'] - SI_data['I_pop']
    SI_data.insert(4, 'S_pop', S_vals)
    
    columns_to_extract = ['date','N','S_pop','I','I_daily','I_pop','D','D_daily','V','R_0','mobility_index']
    return SI_data[columns_to_extract]

def model_params():
    params = {'R_time':14}
    return params

if __name__ == "__main__":
    data = compile_data()
    params = model_params()
    SI_data = extract_SI_data(data, params)
    print(SI_data.loc[962:985])
