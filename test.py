from pysimplelog import Logger

# initialize
l=Logger("log test")

# change log file basename from simplelog to mylog
l.set_log_file_basename("mylog")

# change log file extension from .log to .pylog
l.set_log_file_extension("pylog")

# Add new log types.
l.add_log_type("super critical", name="SUPER CRITICAL", level=200, color='red', attributes=["bold","underline"])
l.add_log_type("wrong", name="info", color='magenta', attributes=["strike through"])
l.add_log_type("important", name="info", color='black', highlight="orange", attributes=["bold"])

# update error log type
l.update_log_type(logType='error', color='pink', attributes=['underline','bold'])

# print logger
print(l, end="\n\n")

# test logging
l.info("I am info, called using my shortcut method.")
l.log("info", "I am  info, called using log method.")

l.warn("I am warn, called using my shortcut method.")
l.log("warn", "I am warn, called using log method.")

l.error("I am error, called using my shortcut method.")
l.log("error", "I am error, called using log method.")

l.critical("I am critical, called using my shortcut method.")
l.log("critical", "I am critical, called using log method.")

l.debug("I am debug, called using my shortcut method.")
l.log("debug", "I am debug, called using log method.")

l.log("super critical", "I am super critical, called using log method because I have no shortcut method.")
l.log("wrong", "I am wrong, called using log method because I have no shortcut method.")
l.log("important", "I am important, called using log method because I have no shortcut method.")

# print last logged messages
print("")
print("Last logged messages are:")
print("=========================")
print(l.lastLoggedMessage)
print(l.lastLoggedDebug)
print(l.lastLoggedInfo)
print(l.lastLoggedWarning)
print(l.lastLoggedError)
print(l.lastLoggedCritical)

# log data
print("")
print("Log random data and traceback stack:")
print("====================================")
l.info("Check out this data", data=list(range(10)))
print("")

# log error with traceback
import traceback
try:
    1/range(10)
except Exception as err:
    l.error('%s (is this python ?)'%err, tback=traceback.extract_stack())