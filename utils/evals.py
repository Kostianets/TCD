import matplotlib.pyplot as plt

def get_evals(model_label, test_metrics, eval_metrics, suffix=""):
    """
    Vytvorí graf porovnávajúci metriky modelu na testovacej a evalučnej množine.
    """

    metrics_names = list(eval_metrics.keys())
    test_values = [test_metrics[m] for m in metrics_names]
    eval_values = [eval_metrics[m] for m in metrics_names]

    x = range(len(metrics_names))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar([i - width/2 for i in x], test_values, width, label='Test Set')
    ax.bar([i + width/2 for i in x], eval_values, width, label='Evaluation Set')
    ax.set_ylabel('Score')
    ax.set_title(f'Test vs Evaluation Metrics ({model_label})')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    
    # Save the plot to disk using a filename that includes the model label
    plt.savefig(f"models/evaluation_metrics_{model_label}{suffix}.png")
    plt.close()