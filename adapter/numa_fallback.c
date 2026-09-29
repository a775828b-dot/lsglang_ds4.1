// Service-local NUMA fallback: retain singleton locality without hard-node OOM.
#include <linux/mempolicy.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdatomic.h>
static _Atomic unsigned long conversions;
static int relaxed(int mode, const unsigned long *mask, unsigned long maxnode) {
  if (mode != MPOL_BIND || !mask || !maxnode || maxnode > 1024) return mode;
  unsigned count = 0;
  for (unsigned long bit = 0; bit < maxnode; ++bit)
    count += (mask[bit / (8 * sizeof(unsigned long))] >> (bit % (8 * sizeof(unsigned long)))) & 1UL;
  if (count != 1) return mode;
  if (atomic_fetch_add(&conversions, 1) == 0) {
    const char message[] = "[numa_fallback] singleton BIND uses PREFERRED\n";
    ssize_t written = write(STDERR_FILENO, message, sizeof(message) - 1);
    (void)written;
  }
  return MPOL_PREFERRED;
}
long set_mempolicy(int mode, const unsigned long *mask, unsigned long maxnode) {
  return syscall(SYS_set_mempolicy, relaxed(mode, mask, maxnode), mask, maxnode);
}
long mbind(void *start, unsigned long length, int mode, const unsigned long *mask,
           unsigned long maxnode, unsigned flags) {
  return syscall(SYS_mbind, start, length, relaxed(mode, mask, maxnode), mask, maxnode, flags);
}
unsigned long numa_fallback_count(void) { return atomic_load(&conversions); }

// Bundled libnuma may bind its internal symbols locally; intercept LK's public calls too.
#include <dlfcn.h>
#include <sys/mman.h>
#include <errno.h>
void *numa_alloc_onnode(size_t size, int node) {
  void *(*original)(size_t, int) = dlsym(RTLD_NEXT, "numa_alloc_onnode");
  if (!original) { errno = ENOSYS; return NULL; }
  void *result = original(size, node);
  if (result && node >= 0 && node < 1024) {
    unsigned long mask[1024 / (8 * sizeof(unsigned long))] = {0};
    mask[node / (8 * sizeof(unsigned long))] = 1UL << (node % (8 * sizeof(unsigned long)));
    if (mbind(result, size, MPOL_BIND, mask, 1024, 0) < 0) {
      int saved = errno; munmap(result, size); errno = saved; return NULL;
    }
  }
  return result;
}
void numa_bind(void *mask) {
  void (*original)(void *) = dlsym(RTLD_NEXT, "numa_bind");
  if (!original) _exit(127);
  original(mask);
  int mode = 0;
  unsigned long nodes[1024 / (8 * sizeof(unsigned long))] = {0};
  if (syscall(SYS_get_mempolicy, &mode, nodes, 1024, NULL, 0) < 0 ||
      set_mempolicy(mode, nodes, 1024) < 0) _exit(127);
}
