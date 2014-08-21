#include "print_utils.h"
#include <map>
namespace CPlusPlusWilsonDslash { 

  std::map<unsigned long, unsigned long>  buffer_alloc_map;
  std::map<unsigned long, unsigned long>  aligned_alloc_map;




  void masterPrintf(const char * format, ... )
  {

#ifdef QMP_COMMS
    if( QMP_is_primary_node() ) { 
#endif
      va_list args;
      va_start(args,format);
      vprintf(format, args);
      va_end(args);
      fflush(stdout);
#ifdef QMP_COMMS
    }
#endif

  }

  void localPrintf(const char * format, ... )
  {

#ifdef QMP_COMMS
    int size = QMP_get_number_of_nodes();
    int rank = QMP_get_node_number();
#else
    int size = 1;
    int rank = 0;
#endif
    printf("Rank %d of %d: ", rank, size);
    va_list args;
    va_start(args,format);
    vprintf(format, args);
    va_end(args);
  }

};
