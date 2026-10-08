import matplotlib.pyplot as plt

def set_plot_style():
    """
    Applies the project palette and styling to matplotlib.
    """
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = '#FFFDF3' # cream-50
    plt.rcParams['axes.facecolor'] = '#FFFDF3'
    plt.rcParams['savefig.facecolor'] = '#FFFDF3'
    plt.rcParams['text.color'] = '#2A2438'       # ink-900
    plt.rcParams['axes.labelcolor'] = '#2A2438'
    plt.rcParams['xtick.color'] = '#2A2438'
    plt.rcParams['ytick.color'] = '#2A2438'
    
    # No top or right spines
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    # Save resolution
    plt.rcParams['savefig.dpi'] = 200

# Named colours for the three model series
COLORS = {
    'custom_cnn': '#7C5FA6',        # purple-500
    'resnet50_finetuned': '#C39A45', # ochre
    'resnet50_frozen': '#B3ACDF'     # lavender-300
}
