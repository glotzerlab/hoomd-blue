// implemented after
// http://www.justsoftwaresolutions.co.uk/threading/implementing-a-thread-safe-queue-using-condition-variables.html

#ifndef __WORK_QUEUE_H__
#define __WORK_QUEUE_H__

#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>
#include <queue>

template<typename Data>
class WorkQueue
{
    public:
        /*!\param limit Limit for queue size (0=no limit)
         */
        WorkQueue(unsigned int limit=0)
            : m_limit(limit)
            { }

        //! Add work to the queue
        void push(Data const& data)
        {
            boost::mutex::scoped_lock lock(m_mutex);
            while (m_limit && m_queue.size() >= m_limit)
                {
                m_below_limit.wait(lock);
                }

            m_queue.push(data);
            lock.unlock();
            m_condition_variable.notify_one();
        }

        bool empty() const
        {
            boost::mutex::scoped_lock lock(m_mutex);
            return m_queue.empty();
        }

        bool try_pop(Data& popped_value)
        {
            boost::mutex::scoped_lock lock(m_mutex);
            if(m_queue.empty())
            {
                return false;
            }
            
            popped_value=m_queue.front();
            m_queue.pop();
            if (m_limit && m_queue.size() < m_limit)
                m_below_limit.notify_one();

            return true;
        }

        Data wait_and_pop()
        {
            boost::mutex::scoped_lock lock(m_mutex);
            while(m_queue.empty())
                {
                m_condition_variable.wait(lock);
                }
            
            Data popped_value=m_queue.front();
            m_queue.pop();
            if (m_limit && m_queue.size() < m_limit)
                m_below_limit.notify_one();
            return popped_value;
        }

    private:
        std::queue<Data> m_queue;
        mutable boost::mutex m_mutex;
        boost::condition_variable m_condition_variable;
        boost::condition_variable m_below_limit;
        unsigned int m_limit;
};

#endif // __WORK_QUEUE_H
