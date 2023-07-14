import matplotlib.pyplot as plt
import seaborn as sns


def Plots(data):
    # visualization Additional_Number_of_Scoring column
    sns.set(style="darkgrid")
    sns.histplot(data=data, x=data["Additional_Number_of_Scoring"])
    plt.show()

    # visualization Total_Number_of_Reviews column
    sns.set(style="darkgrid")
    sns.histplot(data=data, x=data["Total_Number_of_Reviews"])
    plt.show()

    # visualization Average_Score column
    sns.set(style="darkgrid")
    sns.histplot(data=data, x=data["Average_Score"])
    plt.show()

    # visualization lat column
    sns.set(style="darkgrid")
    sns.histplot(data=data, x=data["lat"])
    plt.show()

    # visualization lng column
    sns.set(style="darkgrid")
    sns.histplot(data=data, x=data["lng"])
    plt.show()

