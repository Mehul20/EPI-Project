from mobility_model import clean_data
import matplotlib.pyplot as plt

def plot_data(data):
    for year in [2020, 2021, 2022]:
        current_year_data = data[data['year'] == year]
        
        _, axis_left = plt.subplots(figsize=(12, 6))

        axis_left.plot(current_year_data["week"], current_year_data["people_fully_vaccinated"], label="People Fully Vaccinated", color = "green")
        axis_left.set_ylabel("People Fully Vaccinated", color = "green")
        axis_left.set_title("People Fully Vaccinated and New Cases for Year - " + str(year))

        axis_right = axis_left.twinx()
        axis_right.plot(current_year_data['week'], current_year_data["new_cases"], label="New Cases", color = "red")
        
        axis_right.set_ylabel("New Cases", color = "red")

        axis_left.grid(True)
        axis_left.set_xlabel("Week")

        plt.savefig("../plots/vaccination/" + str(year) + "-cases-plot.png", format = "png")
        plt.show()

if __name__ == "__main__":
    data = clean_data()
    data["people_fully_vaccinated"] = data["people_fully_vaccinated"].fillna(0)
    plot_data(data)

