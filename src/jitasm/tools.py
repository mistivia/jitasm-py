def require(cond):
    if not cond:
        raise AssertionError('requirement not satified')

def ensure(expr):
    def deco(f):
        def newf(*args, **kwargs):
            result = f(*args, **kwargs)
            require(expr(result))
            return result
        return newf
    return deco
