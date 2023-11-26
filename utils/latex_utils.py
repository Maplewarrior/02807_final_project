def createLatexTable(score_metrics: dict[str, dict[str, dict[str, float]]], 
                       caption="Experiment results.", 
                       label="tab:results-table",
                       number_of_decimal_points: int = 4,
                       divide_factor: int = 1):
    """Prints a latex table from the scores and times dictionaries.
    
    Args:
        score_metrics (dict): A dictionary containing the scores for each model and dataset
        caption (str, optional): Caption of the table. Defaults to "Experiment results.".
        label (str, optional): Label of the table. Defaults to "tab:results-table".
    
    Returns:
        None
    """
    print("\\begin{table}[h]")
    print("\\begin{tabular}{ll|lllll}")
    print("\\textbf{Dataset} & \\textbf{Models} & \\textbf{Time} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{MRR} \\\\ \\hline")

    for dataset in score_metrics.keys():
        # Loop through scores for all the models 
        for i, model_name in enumerate(score_metrics[dataset].keys()):
            precision = round(score_metrics[dataset][model_name][("precision", divide_factor)], number_of_decimal_points)
            recall = round(score_metrics[dataset][model_name][("recall", divide_factor)], number_of_decimal_points)
            reciprocal_rank = round(score_metrics[dataset][model_name][("reciprocal_rank", divide_factor)], number_of_decimal_points)
            time = round(score_metrics[dataset][model_name]["time"], number_of_decimal_points)

            if i==0:
                stri = f"\multirow{{{len(score_metrics[dataset].keys())}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{dataset}}}}} & {model_name} & {time} s & {precision} & {recall} & {reciprocal_rank} \\\\"
            else:
                stri = f" & {model_name} & {time} s & {precision} & {recall} & {reciprocal_rank} \\\\"
            print(stri)    
        
        print("\\hline")

    print("\\end{tabular}")
    print(f"\\caption{{{caption}}}")
    print(f"\\label{{{label}}}")
    print("\\end{table}")