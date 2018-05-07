#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 08:14:58 2018

Still rubbish, just experimenting a bit while reading the documentation.

@author: Theo
"""

import sqlite3 as sql3
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('__name__')

    
if __name__ == "__main__":
    
    logger.info('Start reading databse')

    logger.info('Opening test.db')
    
    con = sql3.connect('test.db')
    
    np
    
    with con:
        cur = con.cursor()
        #version = cur.execute('SELECT SQLITE_VERSION')
        logger.info('sqlite version {}'.format(sql3.version))
        data = cur.fetchone()
        records = { 'John': 56, 'Mary': 23}
        logger.debug('records: {}'.format(records))
        logger.info('records updated')

    