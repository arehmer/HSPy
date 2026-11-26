# -*- coding: utf-8 -*-
"""
LuT.from_table_number() and the drivers' load_lut(), checked without a sensor.

Run with:
    python -m pytest tests/test_lut_by_table_number.py
"""

import pytest

from hspy.LuT import LuT, LUT_FOLDER
from hspy.drivers.i2c import I2C_HTPA32x32d
from hspy.drivers.spi import SPI_HTPA160x120dR1


def test_table_in_luts_folder_is_found():
    lut = LuT.from_table_number(150)

    assert lut.sheet_name == '60x40dL1k9'
    assert lut.meta['Tablenumber'] == 150
    assert lut.LuT.shape == (1588, 12)
    assert lut.LuTGridInterpolator is not None


def test_missing_table_gives_None():
    assert LuT.from_table_number(99999) is None


def test_ambiguous_table_number_raises():
    # 113 exists as _v1, _v2 and _v3
    with pytest.raises(ValueError, match='LuTNr113_'):
        LuT.from_table_number(113)


def test_luts_folder_is_the_package_folder():
    assert LUT_FOLDER.name == 'luts' and LUT_FOLDER.parent.name == 'hspy'


@pytest.mark.parametrize('driver_cls', [I2C_HTPA32x32d, SPI_HTPA160x120dR1])
@pytest.mark.parametrize('table_number, applyCalib, calcdK',
                         [(150, True, True),
                          (150, False, False),
                          (99999, True, False)])
def test_load_lut_sets_calcdK(driver_cls, table_number, applyCalib, calcdK):
    # A driver without hardware: skip __init__ (it opens the bus) and set
    # only what load_lut() uses
    driver = driver_cls.__new__(driver_cls)
    driver._calib = {'table_number': table_number}
    driver.applyCalib = applyCalib

    driver.load_lut()

    assert driver.calcdK is calcdK
    assert (driver.LuT is not None) is (table_number == 150)


@pytest.mark.parametrize('driver_cls', [I2C_HTPA32x32d, SPI_HTPA160x120dR1])
def test_assigning_a_LuT_switches_calcdK_on(driver_cls):
    driver = driver_cls.__new__(driver_cls)
    driver._calib = {'table_number': 99999}
    driver.applyCalib = True
    driver.load_lut()
    assert driver.calcdK is False

    driver.LuT = LuT.from_table_number(150)

    assert driver.calcdK is True
