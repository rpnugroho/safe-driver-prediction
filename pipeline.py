from numpy import nan
import config as cfg
from sklearn.pipeline import Pipeline
from skpdspipe.pipeline import DFColumnsSelector, DFFeatureUnion
from skpdspipe.impute import DFSimpleImputer
from skpdspipe.apply import DFSetDType, DFStringify
from skpdspipe.encode import DFDummyEncoder


binary_pipe = Pipeline([
    ('slctr', DFColumnsSelector(cfg.BINARY_FEATURES)),
    ('imptr', DFSimpleImputer(missing_values=-1, fill_value=-9)),
    ('dtype', DFSetDType('int'))
])

categorical_pipe = Pipeline([
    ('slctr', DFColumnsSelector(cfg.CATEGORICAL_FEATURES)),
    # ('oncdr', DFDummyEncoder())
    ('float', DFSetDType('float')),
    ('imprt', DFSimpleImputer(missing_values=-1,
                              fill_value=nan)),
    ('dtype', DFSetDType('category'))
    # ('strng', DFStringify())
])

numerical_pipe = Pipeline([
    ('slctr', DFColumnsSelector(cfg.NUMERICAL_FEATURES)),
    ('imptr', DFSimpleImputer(missing_values=-1, fill_value=-9)),
    ('dtype', DFSetDType('float'))
])


clean_pipe = DFFeatureUnion([
    ('bin', binary_pipe),
    ('cat', categorical_pipe),
    ('num', numerical_pipe)
])
