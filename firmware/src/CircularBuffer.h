#ifndef MBK_CIRCULAR_BUFFER_H
#define MBK_CIRCULAR_BUFFER_H

#include <Arduino.h> 


template <typename T, size_t N>
struct CircularBuffer {
  const int capacity = N;
  T buffer[N]{};
  int count = 0;
  T* head;
  T* tail;

  CircularBuffer(): head(buffer), tail(buffer) {}

  bool isEmpty() { return count == 0; }
  bool isFull() { return count == capacity; }
  int size() { return count; }
  
  void clear() {
    head = buffer;
    tail = buffer;
    count = 0;
  }
  
  T* getFirst() { 
      if (isEmpty()) return nullptr;
    T* oldest = head;
    if (++head == buffer + capacity) {
      head = buffer;
    }
    count--;
    return oldest;
   }

   bool pop(T& out){
    if(isEmpty()) return false;

    out = *head;
    if(++head == buffer + capacity) head = buffer;
    count --;
    return true;
   }
  


/**
 * @brief Pushes a T onto the buffer.
 *
 * This function adds a T to the buffer. If the buffer is full,
 * the oldest T is overwritten.
 *
 * @param cmd A pointer to the T string to be added.
 * @param len The length of the T string.
 * @return True if the T was successfully added, false if the buffer was full and the oldest T was overwritten.
 */

  bool push(const char *cmd, int len) {
    if (++tail == buffer + capacity) {
      tail = buffer;
    }
    memcpy(tail->data, cmd, len);
    tail->len = len;

    if (count == capacity) {
      if (++head == buffer + capacity) {
        head = buffer;
      }
      return false;
    } else {
      if (count++ == 0) {
        head = tail;
      }
      return true;
    }
  }
};


#endif