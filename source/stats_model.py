# calculate power curves for varying sample and effect size
# parameters for power analysis
from numpy import array
from matplotlib import pyplot
from statsmodels.stats.power import TTestIndPower

effect = 0.8
alpha = 0.05
power = 0.8


if __name__ == '__main__':
    # parameters for power analysis
    effect_sizes = array([0.2, 0.5, 0.8])
    sample_sizes = array(range(5, 100))
    # calculate power curves from multiple power analyses
    analysis = TTestIndPower()
    result = analysis.solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha)
    print('Sample Size: %.3f' % result)
    analysis.plot_power(dep_var='nobs', nobs=sample_sizes, effect_size=effect_sizes)
    pyplot.show()
    
    
    


