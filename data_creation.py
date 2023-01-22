from pathlib import Path
from urllib import request
import flickrapi

key = u'05fb00bf76508cebe1dc6ec032fc7685'
secret = u'8dbd5e11cbf33512'
flickr = flickrapi.FlickrAPI(key, secret)

animal_list = ['bird', 'squirrel']

print('Loading...')

for animal in animal_list:
    photos = flickr.walk(
        text=animal,
        extras='url_c',
    )        

    for count, photo in enumerate(photos):
        pathname = Path.cwd() / f"images/{animal}"
        picurl = photo.get('url_c')

        try:
            if len(list(pathname.glob('*.jpg'))) < 500:
                if count == 0:
                    Path.mkdir(pathname, parents=True)
                request.urlretrieve(picurl, pathname / f"{count + 1}.jpg")
            else:
                break
        except:
            pass

print('Done!')