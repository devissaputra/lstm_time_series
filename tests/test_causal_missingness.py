import ast
from pathlib import Path
import unittest
import pandas as pd

class CausalMissingnessTests(unittest.TestCase):
    def test_loader_does_not_borrow_a_future_measurement(self):
        # Isolate the actual loader from the optional PyTorch model dependency.
        tree=ast.parse((Path(__file__).parents[1]/'src/run_experiment.py').read_text())
        loader=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_weekly_series')
        class Dataset:
            def load_pandas(self):
                class Frame: pass
                out=Frame()
                out.data=pd.DataFrame({'co2':[1.0,None,100.0]},index=pd.date_range('2020-01-05',periods=3,freq='W'))
                return out
        namespace={'co2':Dataset()}
        exec(compile(ast.Module(body=[loader],type_ignores=[]),'<loader>','exec'),namespace)
        result=namespace['load_weekly_series']()
        self.assertEqual(result.iloc[1],1.0)
