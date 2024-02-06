def get_task(args):
    if args.task == '20q':
        from uot.tasks.twenty_question import Q20Task
        return Q20Task(args)
    elif args.task == 'md':
        from uot.tasks.medical_diagnosis import MDTask
        return MDTask(args)
    elif args.task == 'tb':
        from uot.tasks.troubleshooting import TBTask
        return TBTask(args)
    else:
        raise NotImplementedError
