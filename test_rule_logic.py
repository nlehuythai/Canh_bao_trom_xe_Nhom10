import unittest

from main import AlertEngine, Detection


class AlertEngineTests(unittest.TestCase):
    def test_prolonged_proximity_triggers_review_alert(self):
        engine = AlertEngine(alert_seconds=3.0, cooldown_seconds=30.0)
        detections = [
            Detection("person", 1, (100, 100, 160, 240)),
            Detection("motorcycle", 2, (175, 150, 300, 245)),
        ]
        events = []
        for frame in range(40):
            new_events, _ = engine.update(detections, frame / 10.0, 640)
            events.extend(new_events)
        self.assertGreaterEqual(len(events), 1)
        self.assertEqual(events[0]["person_id"], 1)
        self.assertEqual(events[0]["motorcycle_id"], 2)

    def test_no_alert_when_objects_are_far_apart(self):
        engine = AlertEngine(alert_seconds=1.0)
        events = []
        for frame in range(30):
            detections = [
                Detection("person", 1, (10, 10, 50, 150)),
                Detection("motorcycle", 2, (400, 200, 540, 300)),
            ]
            new_events, _ = engine.update(detections, frame / 10.0, 640)
            events.extend(new_events)
        self.assertEqual(events, [])

    def test_moving_motorcycle_near_person_triggers_review_alert(self):
        engine = AlertEngine(alert_seconds=8.0, moving_ratio=0.03)
        events = []
        for frame in range(35):
            t = frame / 10.0
            x = 200 + (max(0.0, t - 1.0) * 60.0)
            person = Detection("person", 1, (x - 90, 120, x - 25, 300))
            bike = Detection("motorcycle", 2, (x, 210, x + 130, 315))
            new_events, _ = engine.update([person, bike], t, 960)
            events.extend(new_events)
        self.assertTrue(any("moves" in e["reason"] for e in events))


if __name__ == "__main__":
    unittest.main(verbosity=2)
