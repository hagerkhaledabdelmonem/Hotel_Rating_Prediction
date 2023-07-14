import matplotlib.pyplot as plt

def Plots(data):

    plt.style.use('bmh')
    a = data['Average_Score']
    b = data['Reviewer_Score']
    plt.xlabel('Average_Score')
    plt.ylabel('Reviewer_Score')
    plt.scatter(a,b , c = 'green' , marker ='*')
    plt.show()

    plt.style.use('bmh')
    a = data['Additional_Number_of_Scoring']
    b = data['Reviewer_Score']
    plt.xlabel('Additional_Number_of_Scoring')
    plt.ylabel('Reviewer_Score')
    plt.scatter(a,b , c = 'green' , marker ='*')
    plt.show()

    plt.style.use('bmh')
    a = data['Total_Number_of_Reviews']
    b = data['Reviewer_Score']
    plt.xlabel('Total_Number_of_Reviews')
    plt.ylabel('Reviewer_Score')
    plt.scatter(a,b , c = 'green' , marker ='*')
    plt.show()