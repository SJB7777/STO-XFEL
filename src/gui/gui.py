import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.optimize import curve_fit
from preprocess.preprocess import get_linear_regression_confidence_lower_upper_bound

def find_outliers_gui(y, x):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9)
    
    # Pre-calculate data for performance
    sigma_range = np.arange(0.1, 20.1, 0.1)
    lower_bounds = []
    upper_bounds = []
    y_fits = []
    for sigma in sigma_range:
        lb, ub, yf = get_linear_regression_confidence_lower_upper_bound(y, x, sigma)
        lower_bounds.append(lb)
        upper_bounds.append(ub)
        y_fits.append(yf)
    
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    y_fits = np.array(y_fits)

    # Initial plot
    sigma_init = 3.0
    idx = int((sigma_init - 0.1) / 0.1)
    within_bounds = (y >= lower_bounds[idx]) & (y <= upper_bounds[idx])
    
    normal_points, = ax.plot(x[within_bounds], y[within_bounds], 'o', color='blue', markersize=3, label='Normal Points')
    outlier_points, = ax.plot(x[~within_bounds], y[~within_bounds], 'o', color='red', markersize=3, label='Outliers')
    line_fit, = ax.plot(x, y_fits[idx], color='black', label='Fitted Line')
    fill_between = ax.fill_between(x, lower_bounds[idx], upper_bounds[idx], color='gray', alpha=0.2, label='Confidence Interval')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Linear Regression with Confidence Interval for Outlier Detection')
    y_range = np.max(y) - np.min(y)
    ax.set_ylim([np.min(y) - y_range * 0.1, np.max(y) + y_range * 0.1])
    ax.legend(loc='lower right')

    # Add sigma value text
    sigma_text = ax.text(0.02, 0.98, f'Sigma: {sigma_init:.1f}', transform=ax.transAxes, verticalalignment='top')

    # Add outlier count text
    outlier_count = np.sum(~within_bounds)
    outlier_text = ax.text(0.02, 0.93, f'Outliers: {outlier_count} ({outlier_count/len(y):.1%})', 
                           transform=ax.transAxes, verticalalignment='top')

    axsigma = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    sigma_slider = Slider(axsigma, 'Sigma', 0.1, 20.0, valinit=sigma_init, valstep=0.1)

    def update(val):
        sigma = sigma_slider.val
        idx = int((sigma - 0.1) / 0.1)  # Convert sigma to index
        line_fit.set_ydata(y_fits[idx])
        
        # Clear previous fill_between collection
        for coll in ax.collections:
            if isinstance(coll, plt.matplotlib.collections.PolyCollection):
                coll.remove()
        
        ax.fill_between(x, lower_bounds[idx], upper_bounds[idx], color='gray', alpha=0.2)
        
        within_bounds = (y >= lower_bounds[idx]) & (y <= upper_bounds[idx])
        normal_points.set_data(x[within_bounds], y[within_bounds])
        outlier_points.set_data(x[~within_bounds], y[~within_bounds])
        
        # Update sigma text
        sigma_text.set_text(f'Sigma: {sigma:.1f}')
        
        # Update outlier count
        outlier_count = np.sum(~within_bounds)
        outlier_text.set_text(f'Outliers: {outlier_count} ({outlier_count/len(y):.1%})')
        
        fig.canvas.draw_idle()

    sigma_slider.on_changed(update)

    plt.show()

    return round(sigma_slider.val, 1)


if __name__ == "__main__":
    # Example data
    np.random.seed(0)  # For reproducibility
    x = np.random.normal(10, 5, 100)
    x.sort()
    y = 2.5 * x + np.random.normal(0, 2, 100)
    y[0] += 10  # Add an outlier
    y[-1] -= 10  # Add another outlier

    final_sigma = find_outliers_gui(y, x)
    print(f"Final Sigma: {final_sigma}")