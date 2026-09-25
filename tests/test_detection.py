import importlib.util
from pathlib import Path
import unittest
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'your-detection.py'
spec = importlib.util.spec_from_file_location('mac_detection', SCRIPT)
app = importlib.util.module_from_spec(spec)


class DetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if SCRIPT.exists():
            spec.loader.exec_module(app)

    def test_class_ids_come_from_official_label_map(self):
        labels = app.load_labels(SCRIPT.parent / 'models/mscoco_label_map.pbtxt')
        self.assertEqual(labels[1], 'person')
        self.assertEqual(labels[18], 'dog')
        self.assertEqual(labels[90], 'toothbrush')

    def test_converts_normalized_box_to_image_pixel_coordinates(self):
        # A mistaken axis order or failure to scale by image dimensions must fail.
        record = app.detection_record(
            class_id=18, name='dog', confidence=.75,
            normalized_box=(.2, .1, .8, .6), image_width=500, image_height=300)
        self.assertEqual(record, {
            'ClassID': 18, 'ClassName': 'dog', 'Confidence': .75,
            'Left': 100.0, 'Top': 30.0, 'Right': 400.0, 'Bottom': 180.0,
            'Width': 300.0, 'Height': 150.0, 'Area': 45000.0,
            'Center': [250.0, 105.0]})

    def test_rejects_invalid_geometry(self):
        with self.assertRaises(ValueError):
            app.detection_record(18, 'dog', .8, (.1, .8, .5, .2), 500, 300)

    def test_selects_requested_class_only(self):
        rows = [dict(ClassName='person', Confidence=.99),
                dict(ClassName='dog', Confidence=.71),
                dict(ClassName='dog', Confidence=.82)]
        self.assertEqual(app.select_detection(rows, 'dog')['Confidence'], .82)
        self.assertIsNone(app.select_detection(rows, 'cat'))

    def test_annotation_draws_only_selected_detection(self):
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        person = {'ClassName': 'person', 'Confidence': .99,
                  'Left': 10, 'Top': 50, 'Right': 90, 'Bottom': 180}
        dog = {'ClassName': 'dog', 'Confidence': .8,
               'Left': 150, 'Top': 50, 'Right': 280, 'Bottom': 180}
        result = app.annotate(image, [person, dog], dog)
        self.assertTrue(result[50, 150].any())
        self.assertFalse(result[50, 10].any())


if __name__ == '__main__':
    unittest.main()
