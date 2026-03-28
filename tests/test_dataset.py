import os
import unittest
import pandas as pd

class TestDataset(unittest.TestCase):
    def setUp(self):
        # Ruta al dataset crudo
        self.dataset_path = os.path.join("data", "raw", "sdss_sample.csv")

    def test_dataset_exists(self):
        """Prueba si el archivo del dataset existe en la ruta esperada"""
        self.assertTrue(os.path.exists(self.dataset_path), "El dataset no existe en la ruta especificada")

    def test_dataset_can_be_loaded(self):
        """Prueba si el dataset puede ser cargado con pandas"""
        if os.path.exists(self.dataset_path):
            df = pd.read_csv(self.dataset_path)
            self.assertFalse(df.empty, "El dataset está vacío")
            self.assertGreater(len(df.columns), 0, "El dataset no tiene columnas")

    def test_required_columns_exist(self):
        """Prueba si las columnas requeridas existen (u, g, r, i, z, class)"""
        if os.path.exists(self.dataset_path):
            df = pd.read_csv(self.dataset_path)
            required_cols = ["u", "g", "r", "i", "z", "class"]
            for col in required_cols:
                self.assertIn(col, df.columns, f"La columna {col} no está en el dataset")

if __name__ == '__main__':
    unittest.main()
