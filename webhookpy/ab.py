
import functools
hyperlink_format = '<a href="{link}">{text}</a>'
link_text = functools.partial(hyperlink_format.format)
link_text(link='http://foo/bar', text='linky text')


hyperlink_format.format(link='http://foo/bar', text='linky text')
print("linky text")