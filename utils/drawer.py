import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns


class Drawer:
    def __init__(self, df):
        self.df = df

    def plot_churn_dist(self):
        # Рахуємо кількість відтоку та не відтоку
        churn_counts = self.df['Churn'].value_counts()

        # Рахуємо відсоток відтоку та не відтоку
        churn_percentage = self.df['Churn'].value_counts(normalize=True) * 100

        # Побудова графіка
        plt.figure(figsize=(12, 6))

        # Гістограма для кількості
        plt.subplot(1, 2, 1)
        churn_counts.plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title('Number of Churn vs Non-Churn Customers')
        plt.xlabel('Churn Status')
        plt.ylabel('Number of Customers')
        plt.xticks(rotation=0)

        # Паєчарт для відсотків
        plt.subplot(1, 2, 2)
        plt.pie(churn_percentage, labels=churn_percentage.index, autopct='%1.1f%%', colors=['skyblue', 'salmon'])
        plt.title('Percentage of Churn vs Non-Churn Customers')

        plt.tight_layout()
        plt.show()

    def plot_gender_churn_dist(self):
        gender_labels = ['Male', 'Female']
        churn_labels = ['No', 'Yes']
        # Create subplots: use 'domain' type for Pie subplot
        fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
        fig.add_trace(go.Pie(labels=gender_labels, values=self.df['gender'].value_counts(), name="Gender"),
                    1, 1)
        fig.add_trace(go.Pie(labels=churn_labels, values=self.df['Churn'].value_counts(), name="Churn"),
                    1, 2)

        # Use `hole` to create a donut-like pie chart
        fig.update_traces(hole=.45, hoverinfo="label+percent+name", textfont_size=16)

        fig.update_layout(
            title_text="Gender and Churn Distributions",
            # Add annotations in the center of the donut pies.
            annotations=[dict(text='Gender', x=0.18, y=0.5, font_size=25, showarrow=False),
                        dict(text='Churn', x=0.81, y=0.5, font_size=25, showarrow=False)])
        fig.data[0].marker.colors = ('lightblue', 'darkblue')
        fig.data[1].marker.colors = ('lightgreen', 'green')

        fig.show()

    def plot_churn_reason_dist(self):
        # Look at the distribution of reasons for churn
        reason_counts = self.df['ChurnReason'].value_counts().tail(20)

        # Create a figure and axis object
        fig, ax = plt.subplots(figsize=(7, 5))

        # Generate a color palette with a gradient
        colors = sns.color_palette('viridis', len(reason_counts))

        # Plot the bar chart with the color palette
        reason_counts.plot(kind='barh', ax=ax, color=colors)

        # Set the title and labels
        ax.set_title('Distribution of Customer Reasons for Churn')
        ax.set_xlabel('Count')
        ax.set_ylabel('Reason')

        # Invert the y-axis to display the reasons in descending order
        ax.invert_yaxis()

        # Adjust the spacing between the tick labels and the plot
        ax.tick_params(axis='y', pad=8)

        # Remove the grid lines
        ax.grid(False)

        # Show the plot
        plt.show()

    def plot_columns_histogram(self, colum_names):
        fig, axes = plt.subplots(nrows=len(colum_names), ncols=1, figsize=(1, 1))
        colors = ['#256D85', '#FF4040']

        for i, col in enumerate(colum_names):
            ax = axes[i]
            sns.countplot(data=self.df, x=col, hue='Churn', palette=colors, ax=ax)
            ax.set_title(f'Histogram for {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            ax.legend(['No Churn', 'Churn'])
            ax.grid(axis='y')

        plt.tight_layout()
        plt.show()
    
    def plot_boxplots(self):
        self.df.TotalCharges = pd.to_numeric(self.df.TotalCharges, errors='coerce')
        number_features = ['tenure','MonthlyCharges', 'TotalCharges']
        fig, axes = plt.subplots(1, 3, sharey=True, figsize=(20, 4))
        fig.suptitle('Custom Boxplots for Numeric Features', y=1, size=25)
        axes = axes.flatten()
        for i, column in enumerate(number_features):
            sns.boxplot(data=self.df[column], orient='h', ax=axes[i], color='green', legend=False)
            axes[i].set_title(column + ', skewness: ' + str(round(self.df[column].skew(axis=0, skipna=True), 2)))

        plt.tight_layout()
        plt.show()

    def plot_smooth_dist(self):
        sns.set_context('poster', font_scale=0.6)
        fig, ax = plt.subplots(1, 3, figsize=(20, 6))

        ax1 = sns.histplot(x=self.df['tenure'], color='purple', hue=self.df['Churn'], ax=ax[0], bins=15, kde=True, palette='muted')
        ax1.set(xlabel='Tenure', ylabel='Frequency')

        ax2 = sns.histplot(x=self.df['MonthlyCharges'], color='blue', hue=self.df['Churn'], ax=ax[1], bins=15, kde=True, palette='pastel')
        ax2.set(xlabel='Monthly Charges', ylabel='Frequency')

        ax3 = sns.histplot(x=self.df['TotalCharges'], color='green', hue=self.df['Churn'], ax=ax[2], bins=15, kde=True, palette='dark')
        ax3.set(xlabel='Total Charges', ylabel='Frequency')

        plt.tight_layout()
        plt.show()
    
    def plot_correlation(self):
        df_copy = self.df.copy()

        # Переведення значень 'Yes' та 'No' в числові еквіваленти
        df_copy['Churn'] = df_copy['Churn'].map({'Yes': 1, 'No': 0})

        # Обчислення кореляційних коефіцієнтів
        correlation = df_copy.corr(method='pearson')['Churn'].drop('Churn', errors='ignore')

        # Перевірка чи вдалося отримати кореляції
        if correlation.empty:
            print("Стовпець 'Churn' не знайдено або не є числовим.")
        else:
            # Сортування кореляційних коефіцієнтів за зростанням
            correlation = correlation.sort_values()

            # Побудова графіку
            plt.figure(figsize=(10, 6))
            sns.barplot(x=correlation.index, y=correlation.values)
            plt.xticks(rotation=90)
            plt.xlabel('Features')
            plt.ylabel('Correlation with Churn')
            plt.show()