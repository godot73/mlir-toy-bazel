# Bash function for storing piped stuff as dated log files while teeing them
# on the screen.
# - Source this file from ~/.bashrc.
# - Set $LOGTEE_DIR to a directory for log files. If not set, log files are
# stored at /tmp/ by default.
#
# Usage example: If you run the following repeatedly,
#   > ls ~ | logtee foo
#   > ls ~ | logtee foo
#   > ...
#
# The latest log is symlinked as:
#   $LOGTEE_DIR/foo.log
# The older logs are available as:
#   $LOGTEE_DIR/foo-YYYYMMDD-hhmmss

function logtee() {
  DIR=${LOGTEE_DIR:-}
  if [ -z $DIR ]; then
    DIR=/tmp/
  fi
  if [ -z "$1" ]; then
    HEAD=output
  else
    HEAD=$1
  fi
  TIMESTAMP=$(date +%Y%m%d-%H%M%S)
  FILENAME="${DIR}/${HEAD}-${TIMESTAMP}.log"
  LINKNAME="${DIR}/${HEAD}.log"
  ln -s -f $FILENAME $LINKNAME
  tee $FILENAME
}
