from source import compile_data

def clean_data():
    data = compile_data()
    shrink_data = data.groupby(["year", "week"]).agg({
        'new_cases' : 'sum',
        'new_deaths' : 'sum',
        'new_vaccinations': 'sum',
        'year': 'first',
        'week': 'first',
        'avg_USA': 'first'
    })
    shrink_data = shrink_data.rename(columns={'avg_USA': 'mobility_data'})
    shrink_data = shrink_data.dropna(subset=["mobility_data"])
    shrink_data.to_csv('../data/weekly_cleaned_data.csv', index=False)
    return shrink_data

if __name__ == "__main__":
    data = clean_data()