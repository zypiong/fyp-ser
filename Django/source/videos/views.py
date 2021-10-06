from django.shortcuts import render
from .models import Video
from .forms import VideoSearchForm


def VideosView(request):

    searchvalue = ''

    form = VideoSearchForm(request.POST or None)
    if form.is_valid():
        searchvalue = form.cleaned_data.get("search")

    searchresults = Video.objects.filter(name__icontains=searchvalue)

    context = {'form': form,
               'searchresults': searchresults,
               }

    return render(request, 'videos/search_bar.html', context)
