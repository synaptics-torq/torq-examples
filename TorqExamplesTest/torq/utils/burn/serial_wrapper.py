# -*- coding: utf-8 -*-
import queue
import sys
import logging
import threading
import io
import time
from datetime import datetime
import re
from collections import deque
import socket
import subprocess
import os
import random
import binascii
import platform
from contextlib import suppress as ignored
from queue import Queue, Empty

# 3rd Module
import serial
from serial.tools.list_ports import comports
from serial import SerialException

try:
    RE_TYPE = re._pattern_type
except AttributeError:
    RE_TYPE = re.Pattern

DEFAULT_LOGGING_LEVEL = logging.INFO
FIVEBITS = serial.FIVEBITS          # = 5 (int)
SIXBITS = serial.SIXBITS            # = 6 (int)
SEVENBITS = serial.SEVENBITS        # = 7 (int)
EIGHTBITS = serial.EIGHTBITS        # = 8 (int)
PARITY_NONE = serial.PARITY_NONE    # = 'None' (str)
PARITY_EVEN = serial.PARITY_EVEN    # = 'Even' (str)
PARITY_ODD = serial.PARITY_ODD      # = 'Odd' (str)
PARITY_MARK = serial.PARITY_MARK    # = 'Mark' (str)
PARITY_SPACE = serial.PARITY_SPACE  # = 'Space' (str)
STOPBITS_ONE = serial.STOPBITS_ONE  # = 1 (int)
STOPBITS_ONE_POINT_FIVE = serial.STOPBITS_ONE_POINT_FIVE # = 1.5 (float)
STOPBITS_TWO = serial.STOPBITS_TWO  # = 2 (int)
DEFAULTCODING = 'UTF-8'
HEXMODE = 'HEX'
LOCALHOST = '127.0.0.1'
SOCKET_STOPFLAG = b'=!QUIT!='
TIMEOUT = u'Timeout'
MAX_SERIAL_EXCEPTION_TIMES = 3      # Define max retry time if find serial exception
RETRY_GAP = 1                     # When serial exception happen, retry gap time
SERIAL_READ_GAP = 0.001             # Normal serial read gap time
DEFAULT_TIMEOUT = 99999999

class BaseSerialWrapperException(Exception):
    pass

class SerialWrapperException(BaseSerialWrapperException):
    pass

class SerialTimeoutException(BaseSerialWrapperException):
    pass

class BaseHandler(object):
    '''BaseHandler for IOHander/SocketHandler'''
    def __init__(self, logger=None, coding=None):
        self.logger = logger
        self.coding = coding # This define data coding from serial
        assert self.coding is not None

    def __del__(self):
        self.close()

    def update(self, serialthread, data_tuple):
        _, _ = serialthread, data_tuple
        self.logger.critical("Must define update in subclass")
        raise NotImplementedError

    def close(self):
        self.logger.critical("Must define close in subclass")
        raise NotImplementedError

class IOHandler(BaseHandler):
    '''
    This is a object which offer IOHandler for SerialThread.
    Using it when create input_output_blocking/create_logger from SerialThread
    '''
    def __init__(self, filepath=None, logger=None, coding=None, timestamp=True, noread=False):
        '''
        Always create BytesIO for read (without timestmap)
        If define filepath, will use file handler (with or without timestamp)
        Input:
                filepath: filepath, will write UTF-8 for file(str)
                logger: logging-like (should have function debug/info/critical/exception ...)
                timestamp: Control whether write line to file with timestamp or not(bool)
                noread: only logging, not need to read (False)
        '''
        super(IOHandler, self).__init__(logger, coding)
        self.filepath = filepath
        self.timestamp = timestamp
        self.tempdata = u'' # TempData for save data without new line temprarily
        self.temptime = '' # TempTime for save timestamp for TempData
        self.stringlist = deque() # Save all Data as string list
        self.stringcache = u'' # Once user try to get all data, merge all Data from stringlist
        self.noread = noread
        self.lock = threading.Lock()
        if self.filepath:
            try:
                self.file_handler = io.open(self.filepath, 'ab', 0)
                self.logger.debug("Open {} for Serial Output logging".format(self.filepath))
                self.logger.debug("Timestamp: {}".format(self.timestamp))
            except IOError as err:
                self.logger.critical("Open {} Fail".format(self.file_handler.name))
                self.logger.critical("Exception: {!r}".format(err))
                self.logger.exception("Stack: ")
                raise
        self.logger.debug("Create IOHandler({}) Success".format(self))

    def __del__(self):
        self.close()

    def update(self, serialthread, data_tuple):
        '''Get Serial Data/TimeStamp'''
        self.logger.debug("IOHandler Get Update")
        data, data_time = data_tuple
        # if sys.platform == 'win32':
        #     data.replace(u'\r',u'')
        if not self.noread:
            self.lock.acquire()
            self.stringlist.append(data)
            self.lock.release()
        if self.filepath:
            if self.coding == HEXMODE:
                self.write(data, data_time)
            else:
                if self.tempdata:
                    self.logger.debug("Tempdata exist")
                    alldata = self.tempdata + data
                    start = 1 # Write first line with old timestamp (self.temptime)
                else:
                    self.logger.debug("Tempdata not exist")
                    alldata = data
                    start = 0 # Write first line with new timestamp (data_time)
                    self.logger.debug("TempTime Update")
                    self.temptime = data_time
                data_list = alldata.split(u'\n')
                assert len(data_list) >= 1, "alldata is blank"
                if len(data_list) == 1:
                    # There is no new line flag, wait for new data
                    self.logger.debug("There is no new line flag, skip to write line to file")
                    self.tempdata = data_list[0]
                    return
                if self.tempdata:
                    self.logger.debug("Write Tempdata to line with TempTime")
                    self.write(data_list[0], self.temptime)
                last = len(data_list) - 1
                if data_list[-1] == u'':
                    # If last unit is blank, mean there is no new data for tempdata
                    self.logger.debug("AllData with \\n at end, Reset TempData")
                    self.tempdata = u''
                else:
                    self.logger.debug("AllData without \\n at end, Set TempData and TempTime")
                    self.tempdata = data_list[-1]
                    self.temptime = data_time
                assert last >= start, "Invalid last({0}) or start({1})".format(last, start)
                self.logger.debug("Write AllData to file from line {0} to {1}".format(start, last))
                for line in range(start, last):
                    self.write(data_list[line], data_time)
        return

    def write(self, data, timestamp=None):
        '''Write Data to FileHandler'''
        tstr = u"{:%Y-%m-%d %H:%M:%S.%f}".format(timestamp)[:-3] + u' ' if self.timestamp else u''
        if self.coding == HEXMODE:
            write_data = u"{}".format(binascii.hexlify(data).decode('ascii'))
        else:
            write_data = u"{}".format(data)
        write_line = u"{0}{1}{2}".format(tstr, write_data.strip(), os.linesep)
        with self.lock:
            try:
                self.file_handler.write(write_line.encode('UTF-8'))
            except (IOError, ValueError) as err:
                self.logger.critical("Write {} Fail".format(self.file_handler.name))
                self.logger.critical("Exception: {!r}".format(err))
                self.logger.exception("Stack: ")
                raise
            else:
                self.logger.debug("Write data to file successfully")

    def readall(self):
        '''Return all data, if noread is True, return False'''
        if not self.noread:
            blank = b'' if self.coding == HEXMODE else u''
            self.lock.acquire()
            self.stringlist.appendleft(self.stringcache)
            self.stringcache = blank.join(self.stringlist)
            self.stringlist.clear()
            self.lock.release()
            if len(self.stringcache):
                self.logger.debug("Readall: {}".format(len(self.stringcache)))
            return self.stringcache
        return False

    def close(self):
        '''Close FileHandler'''
        try:
            if self.filepath:
                if self.tempdata:
                    self.logger.info("Write rest TempData to file and Reset TempData")
                    self.write(self.tempdata, self.temptime)
                    self.tempdata = u''
                if not self.file_handler.closed:
                    self.file_handler.close()
                    self.logger.info("Close File Handler - {}".format(self.file_handler.name))
        except AttributeError:
            self.logger.info("filepath already GC")
        self.stringcache = u''
        self.stringlist.clear()
        self.tempdata = u''

class SocketHandler(BaseHandler):
    client_socket, client_addr = None, None

    def __init__(self, port, logger=None, coding=None):
        super(SocketHandler, self).__init__(logger, coding)
        self.sleep_time = 0.001
        self.port = str(port)
        self.stopflag = SOCKET_STOPFLAG
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def __del__(self):
        self.close()

    def start(self):
        for _ in range(3):
            try:
                self.socket.bind((LOCALHOST, int(self.port)))
            except socket.error as err:
                self.logger.error("Serial Monitor bind localhost fail")
                time.sleep(1)
                continue
            else:
                break
        else:
            self.logger.error("Serial Monitor bind keep fail")
            raise err

        self.socket.listen(1)
        self.logger.info("Serial Monitor Port: {}".format(self.port))

        while 1:
            if sys.platform == 'win32':
                runtime_l = ['cmd.exe', '/c', 'start', '/HIGH']
            elif platform.dist()[0] in ('debian', 'Ubuntu'):
                runtime_l = ['gnome-terminal', '-e']
            else:
                self.logger.warning("Current Serial Monitor only work for Windows and Ubuntu")
                break
            py_runtime_exe = sys.executable
            monitor_py = os.path.join(os.path.dirname(__file__), 'serial_monitor.py')
            subprocess.Popen(runtime_l + [py_runtime_exe, monitor_py, self.port, self.coding])
            self.client_socket, self.client_addr = self.socket.accept()
            self.logger.info("Serial Monitor Connect")
            self.logger.debug("Socket: {0} | ADDR: {1}".format(self.client_socket, self.client_addr))
            break

    def close(self):
        self.logger.debug("SocketHandler Stop")
        with ignored(socket.error):
            self.client_socket.sendall(self.stopflag)
        with ignored(socket.error):
            self.client_socket.close()

    def update(self, serialthread, data_tuple):
        self.logger.debug("SocketHandler Get Update")
        data, _ = data_tuple
        if self.coding == HEXMODE:
            s_data = binascii.hexlify(data).decode('ascii') + u'\n'
        else:
            s_data = data
        self.client_socket.sendall(s_data.encode(self.coding))

class SerialThread(object):
    '''
    This is a SerialThread which wrapper from pyserial
    It can offer write/wait_for_strings/wait_for_string/create_logger
    '''
    def __init__(self, serial_port, coding=DEFAULTCODING, serial_config=None, console_monitor=True, logger=None, time_zone=None):
        '''
        Init SerialThread, if open serial_port fail, will raise Exception
        Input:
                serial_port: serial port(str)[for Windows, COM1|COM2|... For Linux, /dev/ttyUSB0...]
                coding: define serial data format (str),
                        by default it is UTF-8.
                        Can be like GBK/CJK/HEX(HEX is special mode)
                serial_config: define serial config(dict), format like below (exist value is default):
                                   {'baudrate': 115200, (int)
                                    'bytesize': EIGHTBITS, (int) FIVEBITES/SIXBITS/SEVENBITS/EIGHTBITS
                                    'parity': PARITY_NONE, (str) PARITY_NONE/PARITY_EVEN/PARITY_EVEN/
                                                                 PARITY_ODD/PARITY_MARK/PARITY_SPACE
                                    'stopbits': STOPBITS_ONE, (int/float) STOPBITES_ONE/STOPBITES_ONE_POINT_FIVE/
                                                                          STOPBITES_TWO
                                    'xonxoff': False, (bool)
                                    'rtscts': False, (bool)
                                    'dsrdtr': False, (bool)}
                console_monitor:
                logger: logging object
                time_zone: pytz object, example:pytz.timezone("Etc/GMT"), time zone value, "Asia/Shanghai" for Shanghai time zone, "Etc/GMT" for test board time zone
        '''
        self.time_zone = time_zone
        self.serial_config = serial_config
        self._console_monitor = console_monitor
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging
            self.logger.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                                    level=DEFAULT_LOGGING_LEVEL)
        self._serial = serial.Serial()
        self._serial.port = serial_port
        self.logger.info("Serial Init: Port - {0}({1})".format(self._serial.port, self._serial.name))
        if coding:
            self.coding = coding
        else:
            self.coding = DEFAULTCODING
        self.logger.info("Serial Coding: {0}".format(self.coding))
        self._serial_config = {
            'baudrate': 115200,
            'bytesize': EIGHTBITS,
            'parity': PARITY_NONE,
            'stopbits': STOPBITS_ONE,
            'xonxoff': False,
            'rtscts': False,
            'dsrdtr': False,
            'write_timeout': None,
        }
        if self.serial_config is not None:
            self._serial_config.update(self.serial_config)
        self.logger.info("Serial Config:")
        for key, value in self._serial_config.items():
            self.logger.info("    {0:<15}: {1}".format(key, value))
        self._serial.apply_settings(self._serial_config)
        self._serial_lock = threading.Lock()
        self._serial_q = Queue()
        self._serial_handlers = []
        self._serial_expection_time = 0
        self._alive = False
        self.alive = False
        self._reader_alive = False
        self._notify_alive = False
        self.reader_thread = None
        self.notify_thread = None
        self.logger.info("Serial Init Complete")
        self.start()

    def __del__(self):
        self.close()

    def _start_reader(self):
        '''Start reader thread'''
        self._reader_alive = True
        self.reader_thread = threading.Thread(target=self._reader, name='read')
        self.reader_thread.daemon = True
        self.reader_thread.start()
        self.logger.info("Serial Reader Thread Start")

    def _stop_reader(self):
        '''Stop reader thread only, wait for clean exit of thread'''
        self.logger.debug("Serial Reader Thread Enter Stop")
        if self.reader_thread.is_alive():
            self.logger.debug("Serial Reader Thread Set flag to False")
            self._reader_alive = False
            self.logger.debug("Serial Reader Thread Wait Stop")
            self.reader_thread.join()
            self.logger.info("Serial Reader Thread Stop")
        else:
            self.logger.debug("Serial Reader Thread Already Stop")

    def _reader(self):
        '''loop read data from serial, and notify all handler'''
        try:
            while self._reader_alive:
                self.logger.debug("Clean data and data_d")
                data = b''
                data_d = u''
                with self._serial_lock:
                    try:
                        buff_size = self._serial.in_waiting
                        if self.time_zone:
                            read_time = datetime.now(self.time_zone)
                        else:
                            read_time = datetime.now()
                        if buff_size:
                            self.logger.debug("Get buff_size: {}".format(buff_size))
                            data = self._serial.read(buff_size)
                            data_d = self._decoding(data)
                    except SerialException:
                        self._serial_expection_time += 1
                        self.logger.error("Read Serial Exception. Time: %d", self._serial_expection_time)
                        self.logger.exception("Stack: ")
                        if self._serial_expection_time > MAX_SERIAL_EXCEPTION_TIMES:
                            self.logger.critical("Serial Operation Exception Up to MAX")
                            raise SerialWrapperException
                        self.logger.warning("Try read serial again")
                        time.sleep(RETRY_GAP)
                        continue
                    else:
                        if self._serial_expection_time != 0:
                            self.logger.debug("Clean serial exception time")
                        self._serial_expection_time = 0 # Read success, reset exception time
                if data_d:
                    self.logger.debug("Get data from Serial")
                    if self._serial_handlers:
                        self.logger.debug("Put data to _serial_q")
                        self._serial_q.put((data_d, read_time))
                else:
                    time.sleep(SERIAL_READ_GAP) # Let Serial work slow to reduce CPU if there is no data
        except SerialException:
            self._serial_q.put(None)
            self._reader_alive = False
            self.logger.critical("Read Serial Exception Up to MAX Retry Time: %d", MAX_SERIAL_EXCEPTION_TIMES)
            self.logger.critical("Serial Readable: {}".format(self._serial.readable()))
            self.logger.exception("Stack: ")
            raise SerialWrapperException
        self.logger.info("Reader Thread Stop")
        self._serial_q.put(None)
        self._reader_alive = False

    def _start_notify(self):
        '''Start notify thread'''
        self._notify_alive = True
        self.notify_thread = threading.Thread(target=self._notify, name='notify')
        self.notify_thread.daemon = True
        self.notify_thread.start()
        self.logger.info("Serial Notify Thread Start")

    def _stop_notify(self):
        '''Stop notify thread only, wait for clean exit of thread'''
        self.logger.info("Serial Notify Thread Stop Enter")
        if self.notify_thread.is_alive():
            self.logger.info("Serial Notify Thread Flag to False")
            self._notify_alive = False
            self._serial_q.put(None)
            self.logger.info("Serial Notify Thread wait to stop")
            self.notify_thread.join()
            self.logger.info("Serial Notify Thread Stop")
        else:
            self.logger.debug("Serial Notify Thread Already Stop")

    def _notify(self):
        '''loop read data from queue, and notify all handler'''
        while self._notify_alive:
            self.logger.debug("Try get data from serial_q")
            alldata = self._serial_q.get()
            if alldata:
                self.logger.debug("Get Data")
                for handler in self._serial_handlers:
                    self.logger.debug("Notify Handler: %r", handler)
                    handler.update(self, alldata)
                    self.logger.debug("Handler: handle data down")
            else:
                self.logger.info("Queue Find None, Exit Notify Thread")
                break
        self.logger.info("Notify Thread exit while")
        self._notify_alive = False

    def _add_handler(self, handler):
        if not self._serial_handlers:
            self._start_notify()
        if not handler in self._serial_handlers:
            self._serial_handlers.append(handler)
            self.logger.info("Add Handler({})".format(handler))
        else:
            self.logger.warning("Handler({}) already in handerlist".format(handler))

    def _remove_handler(self, handler):
        with ignored(ValueError):
            self.logger.info("Remove Handler({})".format(handler))
            self._serial_handlers.remove(handler)
        if not self._serial_handlers:
            self.logger.info("No Handler exist, stop Notify Thread")
            self._stop_notify()
            with self._serial_q.mutex:
                self.logger.info("Clean All data in Queue")
                self._serial_q.queue.clear()

    def _encoding(self, data):
        '''Encode Unicode String'''
        if self.coding == HEXMODE:
            assert isinstance(data, bytes)
            return data
        else:
            return data if isinstance(data, bytes) else data.encode(self.coding)

    def _decoding(self, data):
        '''Decode Byte Stream'''
        if self.coding == HEXMODE:
            assert isinstance(data, bytes)
            return data
        else:
            return data.decode(self.coding, 'ignore') if isinstance(data, bytes) else data

    def _write(self, data, q):
        """
        Loop and copy console->serial until self.exit_character character is
        found. When self.menu_character is found, interpret the next key
        locally.
        """
        while True:
            if self._serial.writable():
                try:
                    self._serial_lock.acquire()
                    self._serial.write(self._encoding(data))
                    self._serial.flush()
                except SerialException:
                    self.alive = False
                    self._serial_expection_time += 1
                    self.logger.critical("Write Serial Exception. Time: %d", self._serial_expection_time)
                    self.logger.critical("Serial Writable: {}".format(self._serial.writable()))
                    self.logger.exception("Stack: ")
                    if self._serial_expection_time > MAX_SERIAL_EXCEPTION_TIMES:
                        self.logger.critical("Serial Operation Exception Up to MAX")
                        q.put("Serial Operation Exception Up to MAX")
                    self.logger.warning("Try write serial again")
                    time.sleep(RETRY_GAP)
                    continue
                else:
                    self._serial_expection_time = 0
                    self.logger.info("Write: {!r}".format(data))
                finally:
                    self._serial_lock.release()
            else:
                self.logger.critical("Write Serial Fail because not Writable")
            break

    def start(self):
        '''start worker threads'''
        serial_list = comports()
        if self._serial.port not in (port.device for port in serial_list):
            self.logger.warning("{} not in Exist Serial Ports".format(self._serial.port))
        try:
            self._serial.open()
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()
        except SerialException:
            self.logger.critical("Fail to open {}".format(self._serial.port))
            self.logger.info("Exist Serial Ports:")
            for serial_info in serial_list:
                self.logger.info("    {0.device}: {0.description}|{0.hwid}".format(serial_info))
            raise SerialWrapperException
        else:
            self.alive = True
            self._start_reader()
            self.logger.info("Serial Start Success")

    def stop(self):
        '''Set flag to stop worker threads and remove all exist handler'''
        self.logger.info("Serial Stop Start")
        if self._reader_alive:
            self._stop_reader()
            self.logger.info("Serial Reader Clean")
        if self._notify_alive:
            self._stop_notify()
            templist = self._serial_handlers[:]
            for handler in templist:
                self._remove_handler(handler)
        self.logger.info("Serial Stop Complete")

    def close(self):
        '''Stop all thread and Close Serial'''
        self.logger.info("Serial Close Start")
        self.stop()
        self._serial.close()
        self.logger.info("Serial Close Complete")

    def write(self, data, timeout=99999999):
        if not data:
            self.logger.warning("Write nothing to serial")
            return True
        q = queue.Queue()
        t = threading.Thread(daemon=True, target=self._write, args=(data, q))
        t.start()
        t.join(timeout)
        if not t.is_alive():
            if not q.empty():
                raise SerialWrapperException
        else:
            self.logger.error("write timeout:{}".format(data))
            raise SerialWrapperException

    def send_break(self, duration=0.25):
        '''Send Break to Serial'''
        self._serial_lock.acquire()
        while True:
            try:
                self._serial.send_break(duration)
            except SerialException:
                self._serial_expection_time += 1
                self.logger.critical("Send Break Exception. Time: %d", self._serial_expection_time)
                self.logger.exception("Stack: ")
                if self._serial_expection_time > MAX_SERIAL_EXCEPTION_TIMES:
                    self.logger.critical("Serial Operation Exception Up to MAX")
                    raise SerialWrapperException
                self.logger.warning("Try send break serial again")
                time.sleep(RETRY_GAP)
                continue
            else:
                self._serial_expection_time = 0
                self.logger.info("Send Break Successfully")
            finally:
                self._serial_lock.release()
            break

    @property
    def break_condition(self):
        '''Get Break Condition'''
        return self._serial.break_condition

    @break_condition.setter
    def set_break_condition(self, break_condition):
        '''Set Break Condition'''
        self._serial.break_condition = break_condition

    def input_output_blocking(self, sinput, expect, timeout=None, *args, **kwargs):
        '''
        This function for write serial or callback to check expect list keyword
        Input: sinput (tuple) - (input, input_repeat, input_gap)
                      input:         (str)[Will set func as self.write] /(func)[external function]
                      input_repeat:  (bool)[Used for whether repeat do input]
                      input_gap:     (int/float)[If input_repeat is True, gap time between 2 input]
               expect (tuple) - (expect_list, expect_all)
                      expect_list    (list)[Can be str or re,
                                            for re, no group support for return,
                                            full add full match string]
                      expect_all     (bool)[True for find all keyword in expect_list/False for find one keywrod]
               timeout (int/float)[Timeout for find expect_list]
               *args/**kwargs   [If input is external func, will put *args and **kwargs as func's arg]
        Output: Result (bool)[True for find keyword]
                String (str/list)[If Fail, return Reason. If True, return found keyword list]
                Raw Data from Serial (str)
        '''
        io = IOHandler(logger=self.logger, coding=self.coding)
        self._add_handler(io)
        start_time = time.time()
        if timeout:
            _timeout = timeout
        else:
            _timeout = DEFAULT_TIMEOUT
        if isinstance(sinput[0], (str, bytes)):
            func = lambda: self.write(sinput[0], _timeout)
            name = 'self.write'
            sargs = '{!r}'.format(sinput[0])
        else:
            func = lambda: sinput[0](*args, **kwargs)
            name = sinput[0].__name__
            sargs = u''
            if args:
                sargs = ', '.join((str(x) for x in args))
            if kwargs:
                if args:
                    sargs += ', '
                for key, value in kwargs.items():
                    sargs += u'{0}={1}'.format(key, value)
        input_repeat = sinput[1]
        input_gap = sinput[2]
        expect_list = expect[0][:]
        assert isinstance(expect_list, list)
        exp_gen = (keyword.pattern if isinstance(keyword, RE_TYPE) else keyword for keyword in expect_list)
        kl_s = '|'.join(exp_gen)
        expect_all = expect[1]
        ret = False
        string_found = []
        string_found_count = 0
        self.logger.info("="*40)
        self.logger.info("Input:")
        self.logger.info("    func:      {0}({1})".format(name, sargs))
        self.logger.info("    Repeat:    {}".format(input_repeat))
        self.logger.info("    Gap:       {}".format(input_gap))
        self.logger.info("Expect:")
        self.logger.info("    List:      {}".format(kl_s))
        self.logger.info("    Condition: {}".format(expect_all))
        self.logger.info("Timeout:       {}".format(_timeout))
        self.logger.info("="*40)
        if sinput[0]: # If Input is blank string, skip self.write
            self.logger.info("Run {0}({1})".format(name, sargs))
            func()
        is_loop = True
        while is_loop:
            is_loop = time.time() - start_time < _timeout
            allconsoledata = io.readall()
            if expect_all:
                templist = expect_list[:]
                for keyword in templist:
                    if isinstance(keyword, RE_TYPE):
                        res = keyword.search(allconsoledata)
                        if res:
                            string_found_count += 1
                            string_found.append(res.group(0))
                            expect_list.remove(keyword)
                            self.logger.info("Found: {}".format(res.group(0)))
                            continue
                    elif keyword in allconsoledata:
                        string_found_count += 1
                        string_found.append(keyword)
                        expect_list.remove(keyword)
                        self.logger.info("Found: {}".format(keyword))
                        continue
                self.logger.debug(f"console output: {allconsoledata}")
                if string_found_count == len(expect[0]):
                    ret = True
                    self.logger.info("All Keywords Found")
                    break
            else:
                for keyword in expect_list:
                    if isinstance(keyword, RE_TYPE):
                        res = keyword.search(allconsoledata)
                        if res:
                            ret = True
                            string_found = [res.group(0)]
                            self.logger.info("Found: {}".format(res.group(0)))
                            break
                    elif keyword in allconsoledata:
                        ret = True
                        string_found = [keyword]
                        self.logger.info("Found: {}".format(keyword))
                        break
                self.logger.debug(f"console output: {allconsoledata}")
                if ret:
                    break
            if input_repeat and sinput[0]:
                self.logger.debug("Sleep {}s".format(input_gap))
                time.sleep(input_gap)
                self.logger.info("Run {0}({1})".format(name, sargs))
                func()
            elif is_loop:
                if _timeout != DEFAULT_TIMEOUT:
                    time.sleep(_timeout/10)
                else:
                    time.sleep(1)
        else:
            self.logger.warning("Timed out waiting for the expected string!")
            raise SerialTimeoutException
        self._remove_handler(io)
        allconsoledata = io.readall()
        io.close()
        return ret, string_found, allconsoledata

    def wait_for_strings(self, keywordlist, all_=False, timeout=None):
        '''
        Blocking until find keywordlist
        Input: keywordlist (list) [Can be str or re.compile]
               all (bool) [True for find all keywrodlist, False for find one of keywordlist]
               timeout (float) [Timeout for fail to find keywordlist]
        Output: Result (bool)[True for find keyword]
                String (str/list)[If Fail, return Reason. If True, return found keyword list]
                Raw Data from Serial (str)
        '''
        sinput = (u'', False, 0)
        expect = (keywordlist, all_)
        return self.input_output_blocking(sinput, expect, timeout=timeout)

    def wait_for_string(self, keyword, timeout=None):
        '''
        Blocking until find keyword
        Input: keyword (str/re) [Can be str or re.compile]
               timeout (float) [Timeout for fail to find keyword]
        Output: Result (bool)[True for find keyword]
                String (str/list)[If Fail, return Reason. If True, return found keyword list]
                Raw Data from Serial (str)
        '''
        return self.wait_for_strings([keyword], timeout=timeout)

    def expects_for_write(self, input_data, keywordlist, repeat=False, repeat_gap=1, all_=False, timeout=None):
        '''
        Blocking until find keyword after input to serial
        Input: input_data (byte or str) [Write to serial]
               keywordlist (list) [Can be str or re.compile]
               repeat (bool) [Repeat write input_str to serial]
               repeat_gap (int/float) [Gap between 2 repeat write to serial]
               all (bool) [True for find all keywrodlist, False for find one of keywordlist]
               timeout (float) [Timeout for fail to find keywordlist]
        Output: Result (bool)[True for find keyword]
                String (str/list)[If Fail, return Reason. If True, return found keyword list]
                Raw Data from Serial (str)
        '''
        sinput = (input_data, repeat, repeat_gap)
        expect = (keywordlist, all_)
        return self.input_output_blocking(sinput, expect, timeout=timeout)

    def expect_for_write(self, input_data, keyword, repeat=False, repeat_gap=1, timeout=None):
        '''
        Blocking until find keyword after input to serial
        Input: input_data (byte or str) [Write to serial]
               keyword (str/re) [Can be str or re.compile]
               repeat (bool) [Repeat write input_str to serial]
               repeat_gap (int/float) [Gap between 2 repeat write to serial]
               timeout (float) [Timeout for fail to find keywordlist]
        Output: Result (bool)[True for find keyword]
                String (str/list)[If Fail, return Reason. If True, return found keyword list]
                Raw Data from Serial (str)
        '''
        return self.expects_for_write(input_data, [keyword], repeat, repeat_gap, False, timeout)

    def input_output_blocking_for_TFTP(self, sinput, expect, timeout=None, char_gap=0.01, *args, **kwargs):
        '''
        Compatibility helper for tftp2emmc-style commands.

        The input is written one character at a time to reduce character loss on
        slow consoles, while reusing the normal input/output blocking loop.
        '''
        input_data, repeat, repeat_gap = sinput
        keywordlist, all_ = expect

        def write_char_by_char():
            for char in input_data:
                self.write(char, timeout=timeout if timeout is not None else DEFAULT_TIMEOUT)
                if char_gap:
                    time.sleep(char_gap)

        return self.input_output_blocking((write_char_by_char, repeat, repeat_gap), (keywordlist, all_), timeout=timeout, *args, **kwargs)

    def tftp_to_emmc(self, input_data, keywordlist, repeat=False, repeat_gap=1, all_=False, timeout=None, char_gap=0.01):
        '''
        Blocking write for tftp2emmc-style commands while reading until keywords appear.
        The input is sent one character at a time to avoid dropping characters on slow consoles.
        '''
        self.logger.info(
            "TFTP command: %r, keywords: %s, repeat=%s, repeat_gap=%s, all=%s, timeout=%s",
            input_data,
            ', '.join((kw.pattern if isinstance(kw, RE_TYPE) else str(kw) for kw in keywordlist)),
            repeat,
            repeat_gap,
            all_,
            timeout,
        )
        return self.input_output_blocking_for_TFTP(
            (input_data, repeat, repeat_gap),
            (keywordlist, all_),
            timeout=timeout,
            char_gap=char_gap,
        )

    def create_logger(self, filepath=None, timestamp=True):
        '''
        Create Logger for Serial
        Input: filepath (str)
        Output: filehandler (IOHandler)
        '''
        self.logger.info("Create File Logger: {0} (Timestmap: {1})".format(filepath, timestamp))
        filehandler = IOHandler(filepath=filepath, logger=self.logger, timestamp=timestamp, coding=self.coding)
        self._add_handler(filehandler)
        return filehandler

    def close_logger(self, filehandler):
        '''
        Close/Stop logger for Serial
        Input: filehanderl (IOHandler)
        '''
        self.logger.info("Close Logger: {}".format(filehandler))
        self._remove_handler(filehandler)
        filehandler.close()

    def create_serial_monitor(self, port=None):
        '''
        Create Monitor for Serial
        Input: port (str)[If keep None, will set port random between(10000,65535)]
        Output: sockethandler (IOHandler)
        '''
        self.logger.info("Create Serial Monitor")
        if port:
            sockethandler = SocketHandler(port=port, logger=self.logger, coding=self.coding)
        else:
            sockethandler = SocketHandler(port=random.randint(10000, 65535), logger=self.logger, coding=self.coding)
        sockethandler.start()
        self._add_handler(sockethandler)
        return sockethandler

    def close_serial_monitor(self, sockethandler):
        '''
        Close/Stop logger for Serial
        Input: filehanderl (IOHandler)
        '''
        self.logger.info("Close Monitor: {}".format(sockethandler))
        self._remove_handler(sockethandler)
        sockethandler.close()
