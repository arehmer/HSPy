import hspy

i2c = hspy.drivers.i2c.HTPA32x32d(1)
i2c.open()
i2c.read_status()
