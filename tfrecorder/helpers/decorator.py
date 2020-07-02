PROTO_LIST = 'proto_list'

class tfrecordable:
    """
    This is a wrapper of the real decorator which emulates the @property built-in decorator, but also get a reference
    to the owner of the method decorated. This is because at Example class creation, the function passed to the
    decorator is not yet a method, so has no reference to the owner. However, there is this trick where __set__name__
    is called and passed the owner class reference that we use here to give it a reference to the decorated attributes.

    Args:
        dtype:

    Returns:
        decorator: a Property object.

    """
    def __init__(self, dtype):

        # create a custom Property object with this dtype as class attribute
        class PropertyWithDtype(Property, dtype=dtype):
            pass

        self.property = PropertyWithDtype

    def __call__(self, func):
        return self.property(func)



class MetaProperty(type):
    """
    A metaclass used to custom the creation of the Property class for a given dtype.
    """

    def __new__(mcs, class_name, bases, attrs, dtype=None):

        return super(MetaProperty, mcs).__new__(mcs, class_name, bases, attrs)

    def __init__(cls, class_name, bases, attrs, dtype=None):
        super(MetaProperty, cls).__init__(class_name, bases, attrs)
        cls.dtype = dtype



class Property(metaclass=MetaProperty, dtype=None):
    """
    Creates a descriptor (i.e. settable, gettable and deletable) emulating the built-in @property decorator.
    The only difference is that it keeps track of the dtype that was passed to it in the Example subclass definition
    of the tfrecordables attributes.

    See: https://docs.python.org/3/howto/descriptor.html
    """

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):

        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        if doc is None and fget is not None:
            doc = fget.__doc__
        self.__doc__ = doc

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget(obj)

    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        self.fset(obj, value)

    def __delete__(self, obj):
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        self.fdel(obj)

    def getter(self, fget):
        return type(self)(fget, self.fset, self.fdel, self.__doc__)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel, self.__doc__)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel, self.__doc__)

    def __set_name__(self, owner, name):
        """
        Adds the name and the type of this tfrecordable property to the tfrecordable list of the Example subclass
        instance owning this property.

        Args:
            owner: Example subclass instance
            name: str, the name of this tfrecordable property.
        """
        if hasattr(owner, PROTO_LIST):
            getattr(owner, PROTO_LIST).append((name, self.__class__.dtype))
        else:
            setattr(owner, PROTO_LIST, [(name, self.__class__.dtype)])



