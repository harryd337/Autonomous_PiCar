import tensorflow as tf

class GpuAvailabilityTest(tf.test.TestCase):
  
  def test_gpu_available(self):
    gpu_devices = tf.config.list_physical_devices('GPU')
    self.assertIsNotNone(gpu_devices, "No GPU device detected.")
    
if __name__ == '__main__':
  unittest.main()