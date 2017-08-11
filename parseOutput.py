import sys
import re

fptr = open( sys.argv[1], 'rt' )

enum = 0
tacc = 0.0
vacc = 0.0

while( True ) :
  tmp = fptr.readline()
  if ( tmp == '' ) :
    break

  tmp = tmp.rstrip( "\r\n" )

  m = re.search( "EPOCH", tmp )
  if ( m != None ) :
    m = re.split( " ", tmp )
    enum = m[1]

  m = re.search( "Training Accuracy = ", tmp )
  if ( m != None ) :
    m = re.split( " = ", tmp )
    tacc = m[1]

  m = re.search( "Validation Accuracy = ", tmp )
  if ( m != None ) :
    m = re.split( " = ", tmp )
    vacc = m[1]
    print( enum+" "+tacc+" "+vacc )
    #print( enum+","+tacc+","+vacc )

fptr.close()

