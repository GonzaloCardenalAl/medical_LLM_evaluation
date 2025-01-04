import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_f1_score_average_and_per_question_average(path_to_results,path_to_rephrased_results):

    # Load the data
    df = pd.read_csv('./evaluation_results/evaluation_results_merged.csv')
    df_rephrased = pd.read_csv('./evaluation_results/evaluation_results_rephrased_check_merged.csv')
    
    # Exclude rows where 'category_id' is '3_MCQ'
    df = df[df['category_id'] != '3_MCQ']
    df_rephrased = df_rephrased[df_rephrased['category_id'] != '3_MCQ']
    
    # Prepare data for the first subplot
    grouped_df = df.groupby('category_id')['f1_score_mean'].agg(['mean', 'std']).reset_index()
    grouped_df_rephrased = df_rephrased.groupby('category_id')['f1_score_mean'].agg(['mean', 'std']).reset_index()
    
    # Merge grouped dataframes on 'category_id'
    grouped_merged = pd.merge(
        grouped_df,
        grouped_df_rephrased,
        on='category_id',
        suffixes=('_original', '_rephrased')
    )
    
    # Prepare data for the second subplot
    df_sorted = df.sort_values(by=['category_id', 'question_index']).reset_index(drop=True)
    df_rephrased_sorted = df_rephrased.sort_values(by=['category_id', 'question_index']).reset_index(drop=True)
    
    # Merge df_sorted and df_rephrased_sorted on 'category_id' and 'question_index'
    merged_df = pd.merge(
        df_sorted[['category_id', 'question_index', 'f1_score_mean']],
        df_rephrased_sorted[['category_id', 'question_index', 'f1_score_mean']],
        on=['category_id', 'question_index'],
        suffixes=('_original', '_rephrased'),
        how='inner'
    )
    
    # Initialize variables for the second subplot
    y_positions = []
    y_labels = []
    f1_scores_original = []
    f1_scores_rephrased = []
    colors_list = []
    category_separators = []
    category_labels_positions = []
    category_labels = []
    
    y_pos = 0  # Starting y position
    category_gap = 1  # Gap between categories
    colors = plt.cm.tab20.colors  # Colormap for different categories
    
    categories = merged_df['category_id'].unique()
    
    for i, category in enumerate(categories):
        category_data = merged_df[merged_df['category_id'] == category].reset_index(drop=True)
        num_questions = len(category_data)
        
        # Store position for category label (middle of the category block)
        category_labels_positions.append(y_pos + (num_questions - 1) / 2)
        category_labels.append(f"Category {category}")
        
        # Assign y positions to questions within the category
        for idx in range(num_questions):
            y_positions.append(y_pos)
            y_labels.append(f"Q{int(category_data.iloc[idx]['question_index'])}")
            f1_scores_original.append(category_data.iloc[idx]['f1_score_mean_original'])
            f1_scores_rephrased.append(category_data.iloc[idx]['f1_score_mean_rephrased'])
            colors_list.append(colors[i % len(colors)])
            y_pos += 1  # Move to next position
        
        # Add a separator after each category except the last one
        if i < len(categories) - 1:
            category_separators.append(y_pos - 0.5)
            y_pos += category_gap  # Add gap before the next category
    
    # Set up the figure and axes
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 18))
    plt.subplots_adjust(hspace=0.5)
    
    # First subplot: Average F1 Score by Category ID
    # Plot rephrased data
    axes[0].barh(
        y=grouped_merged['category_id'],
        width=grouped_merged['mean_rephrased'],
        color='white',
        edgecolor='black',
        label='Rephrased'
    )
    
    # Plot original data
    axes[0].barh(
        y=grouped_merged['category_id'],
        width=grouped_merged['mean_original'],
        xerr=grouped_merged['std_original'],
        color='skyblue',
        ecolor='gray',
        capsize=5,
        label='Original'
    )
    
    axes[0].set_xlabel('Average F1 Score')
    axes[0].set_ylabel('Category ID')
    axes[0].set_title('Average F1 Score by Category ID with Standard Deviation')
    axes[0].invert_yaxis()  # Highest category at the top
    axes[0].legend()
    
    # Second subplot: F1 Score by Question Index and Category ID
    # Plot rephrased data bars (background bars)
    axes[1].barh(
        y=y_positions,
        width=f1_scores_rephrased,
        color='white',
        edgecolor='black',
        label='Rephrased'
    )
    
    # Plot original data bars (foreground bars)
    axes[1].barh(
        y=y_positions,
        width=f1_scores_original,
        color=colors_list,
        edgecolor='black',
        label='Original'
    )
    
    # Set y-axis ticks and labels
    axes[1].set_yticks(y_positions)
    axes[1].set_yticklabels(y_labels)
    
    # Add category separators
    for sep in category_separators:
        axes[1].axhline(y=sep, color='black', linewidth=1)
    
    # Add category labels on the left side
    for pos, label in zip(category_labels_positions, category_labels):
        axes[1].text(
            x=-0.08,  # Slightly left of the y-axis (adjust as needed)
            y=pos,
            s=label,
            va='center',
            ha='right',
            fontsize=12,
            transform=axes[1].get_yaxis_transform()
        )
    
    axes[1].set_xlabel('F1 Score')
    axes[1].set_ylabel('Questions')
    axes[1].set_title('F1 Score by Question Index and Category ID')
    axes[1].invert_yaxis()
    axes[1].legend()
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.savefig('f1_scores.jpg')
    plt.show()

def plot_precision_and_recall():
    # Load the dataset
    data = pd.read_csv('./evaluation_results/evaluation_results_merged.csv')
    
    # Filter out rows with category_id '3_MCQ'
    filtered_data = data[data['category_id'] != '3_MCQ']
    
    # Plotting precision and recall with their standard deviations
    plt.figure(figsize=(14, 6))
    
    # Adjusted bar width and increased separation
    bar_width = 0.3
    x = range(len(filtered_data))
    
    # Plotting precision with standard deviation as error bars
    plt.bar(x, filtered_data['precision_mean'], width=bar_width, yerr=filtered_data['precision_std'], 
            label='Precision Mean', capsize=5, align='center')
    
    # Shifting the x positions for recall bars with increased separation
    x_recall = [pos + bar_width for pos in x]
    
    # Plotting recall with standard deviation as error bars
    plt.bar(x_recall, filtered_data['recall_mean'], width=bar_width, yerr=filtered_data['recall_std'], 
            label='Recall Mean', capsize=5, align='center')
    
    # Adding labels and title
    plt.xlabel('Questions')
    plt.ylabel('Score')
    plt.title('Comparison of Precision and Recall with Standard Deviations')
    
    # Setting x-tick labels to show category_id and question_index
    labels = [f"{row['category_id']}-{row['question_index']}" for _, row in filtered_data.iterrows()]
    plt.xticks([pos + bar_width / 2 for pos in x], labels, rotation=90)
    
    plt.legend()
    
    # Display the plot
    plt.tight_layout()
    plt.savefig('Precision_against_recall.jpg')
    plt.show()